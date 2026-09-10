"""CSV analysis and character preprocessing, independent of TensorFlow."""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

PAD, SOS, EOS, UNK = 0, 1, 2, 3


def encode(text, vocabulary, length):
    ids = [vocabulary.get(char, UNK) for char in text.strip()[:length]]
    return ids + [PAD] * (length - len(ids))


def read_data(paths):
    rows, seen, sources = [], set(), []
    for path in map(Path, paths):
        counts = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), rows=0, empty=0, duplicates=0, added=0)
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'Q', 'A'}.issubset(reader.fieldnames or []):
                raise ValueError(f'{path.name}: Q, A 열이 필요합니다.')
            for row in reader:
                counts['rows'] += 1
                pair = ((row.get('Q') or '').strip(), (row.get('A') or '').strip())
                if not all(pair):
                    counts['empty'] += 1
                elif pair in seen:
                    counts['duplicates'] += 1
                else:
                    seen.add(pair)
                    rows.append(dict(Q=pair[0], A=pair[1], source=path.name))
                    counts['added'] += 1
        sources.append(counts)
    if len(rows) < 10:
        raise ValueError('최소 10개 유효 질문·답변이 필요합니다.')
    return rows, sources


def analyze(rows, sources, max_length):
    report = dict(data_sources=sources, count=len(rows), max_length=max_length)
    for field in ['Q', 'A']:
        lengths = np.array([len(row[field]) for row in rows])
        limit = max_length if field == 'Q' else max_length - 1
        report[field] = dict(min=int(lengths.min()), mean=float(lengths.mean()),
                             p50=float(np.percentile(lengths, 50)), p90=float(np.percentile(lengths, 90)),
                             p95=float(np.percentile(lengths, 95)), max=int(lengths.max()),
                             truncated=int((lengths > limit).sum()), limit=limit)
    answers = defaultdict(set)
    for row in rows:
        answers[row['Q']].add(row['A'])
    report['multiple_answer_questions'] = sum(len(values) > 1 for values in answers.values())
    # Surface word frequency, not a morphological or semantic analysis.
    report['top_words'] = {}
    for field in ['Q', 'A']:
        counter = Counter(word.lower() for row in rows for word in re.findall(r'[가-힣A-Za-z0-9]+', row[field]) if len(word) > 1)
        report['top_words'][field] = counter.most_common(30)
    return report


def plot_analysis(rows, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, field, title in zip(axes, ['Q', 'A'], ['Question length', 'Answer length']):
        for source in sorted({row['source'] for row in rows}):
            ax.hist([len(row[field]) for row in rows if row['source'] == source], bins=40, alpha=0.55, label=source)
        ax.set(title=title, xlabel='Characters', ylabel='Number of records')
        ax.legend(fontsize=8)
    fig.savefig(Path(output) / 'length_distribution.png', dpi=140)
    plt.close(fig)


def prepare(paths, output_root, max_length=256, max_samples=None, seed=1234):
    if max_length < 2 or (max_samples is not None and max_samples < 10):
        raise ValueError('max_length >= 2, max_samples >= 10 이어야 합니다.')
    rows, sources = read_data(paths)
    report = analyze(rows, sources, max_length)
    rng = np.random.default_rng(seed)
    # Keep identical (truncated) model inputs in one split to avoid validation leakage.
    groups = defaultdict(list)
    for row in rows:
        groups[row['Q'][:max_length]].append(row)
    keys = list(groups)
    rng.shuffle(keys)
    selected = []
    for key in keys:
        selected.extend(groups[key])
        if max_samples is not None and len(selected) >= max_samples:
            break  # retain whole question groups; sample limit is approximate
    grouped = defaultdict(list)
    for row in selected:
        grouped[row['Q'][:max_length]].append(row)
    if len(grouped) < 2:
        raise ValueError('학습·검증 분리를 위해 서로 다른 질문이 최소 2개 필요합니다.')
    cutoff = max(1, min(len(grouped) - 1, int(len(grouped) * 0.9)))
    train = [row for group in list(grouped.values())[:cutoff] for row in group]
    valid = [row for group in list(grouped.values())[cutoff:] for row in group]
    selected = train + valid
    chars = sorted(set(''.join(r['Q'][:max_length] + r['A'][:max_length - 1] for r in train)))
    vocabulary = {char: i + 4 for i, char in enumerate(chars)}
    x = np.asarray([encode(r['Q'], vocabulary, max_length) for r in selected], dtype='int32')
    targets = [[vocabulary.get(c, UNK) for c in r['A'][:max_length - 1]] + [EOS] for r in selected]
    y = np.asarray([t + [PAD] * (max_length - len(t)) for t in targets], dtype='int32')
    decoder_inputs = np.concatenate([np.full((len(y), 1), SOS, dtype='int32'), y[:, :-1]], axis=1)
    output = Path(output_root) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    config = dict(format_version=1, vocabulary=vocabulary, max_length=max_length, split=len(train),
                  sample_count=len(selected), data_sources=sources, seed=seed)
    np.savez_compressed(output / 'training_data.npz', x=x, y=y, decoder_inputs=decoder_inputs)
    (output / 'preprocess_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    report['training_selection'] = dict(train=len(train), validation=len(valid), vocabulary_size=len(vocabulary)+4,
        truncated_questions=sum(len(r['Q']) > max_length for r in selected),
        truncated_answers=sum(len(r['A']) >= max_length for r in selected))
    (output / 'analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    with (output / 'split_records.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Q', 'A', 'source', 'split'])
        writer.writeheader()
        writer.writerows(dict(row, split='train' if i < len(train) else 'validation') for i, row in enumerate(selected))
    plot_analysis(rows, output)
    for source in sources:
        print(f"{source['file']}: {source['rows']:,}행 / 사용 {source['added']:,} / 중복 {source['duplicates']} / 빈 행 {source['empty']}")
    print(f"통합 {len(rows):,} / 학습 {len(train):,} / 검증 {len(valid):,}")
    print(f"전체 데이터 중 길이 초과: 질문 {report['Q']['truncated']:,} / 답변 {report['A']['truncated']:,}")
    print('저장:', output)
    return output, report


if __name__ == '__main__':
    if Path(sys.prefix).name.lower() != 'esg':
        raise RuntimeError('로컬에서는 esg 가상환경에서 실행하세요.')
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', nargs='+', type=Path, default=[base.parent / 'ChatbotData.csv', base.parent / 'ESG_QnA_dataset_10000.csv'])
    parser.add_argument('--output', type=Path, default=base / 'data_in')
    parser.add_argument('--max-length', type=int, default=256)
    parser.add_argument('--max-samples', type=int)
    args = parser.parse_args()
    prepare(args.data, args.output, args.max_length, args.max_samples)
