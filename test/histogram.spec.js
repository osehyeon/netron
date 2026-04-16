import * as assert from 'assert';
import { test } from 'node:test';

// Histogram computation pure function — mirrors metrics.Tensor.get histogram() logic in view.js
function computeHistogram(values, numBins = 64) {
    if (!values || values.length === 0) return null;

    // Flatten nested arrays, filtering out non-finite values
    const flat = [];
    const stack = [values];
    while (stack.length > 0) {
        const item = stack.pop();
        if (Array.isArray(item)) {
            for (const element of item) {
                stack.push(element);
            }
        } else if (typeof item === 'number' && isFinite(item)) {
            flat.push(item);
        }
    }
    if (flat.length === 0) return null;

    let min = flat[0];
    let max = flat[0];
    for (const v of flat) {
        if (v < min) min = v;
        if (v > max) max = v;
    }

    const counts = new Array(numBins).fill(0);
    const range = max - min;
    if (range === 0) {
        counts[0] = flat.length;
    } else {
        for (const v of flat) {
            const bin = Math.min(Math.floor((v - min) / range * numBins), numBins - 1);
            counts[bin]++;
        }
    }

    const edges = new Array(numBins + 1);
    for (let i = 0; i <= numBins; i++) {
        edges[i] = min + (range * i) / numBins;
    }

    let sum = 0;
    for (const v of flat) sum += v;
    const mean = sum / flat.length;
    let variance = 0;
    for (const v of flat) variance += (v - mean) * (v - mean);
    const std = Math.sqrt(variance / flat.length);

    return { counts, edges, min, max, mean, std, total: flat.length };
}

// Helper: sum of counts array
function sumCounts(counts) {
    return counts.reduce((a, b) => a + b, 0);
}

test('normal case: counts sum equals total', () => {
    const values = [1.0, 2.5, 3.0, 4.5, 5.0, 0.5, 2.0, 3.5];
    const hist = computeHistogram(values);
    assert.ok(hist !== null, 'histogram should not be null');
    assert.strictEqual(sumCounts(hist.counts), hist.total);
    assert.strictEqual(hist.total, values.length);
});

test('normal case: min and max are correct', () => {
    const values = [1.0, 2.5, 3.0, 4.5, 5.0, 0.5, 2.0, 3.5];
    const hist = computeHistogram(values);
    assert.strictEqual(hist.min, 0.5);
    assert.strictEqual(hist.max, 5.0);
});

test('normal case: mean calculation is correct', () => {
    const values = [1.0, 2.0, 3.0, 4.0, 5.0];
    const hist = computeHistogram(values);
    const expectedMean = (1 + 2 + 3 + 4 + 5) / 5;
    assert.ok(Math.abs(hist.mean - expectedMean) < 1e-10, `mean ${hist.mean} should be ${expectedMean}`);
});

test('normal case: edges length is counts.length + 1', () => {
    const values = [1.0, 2.0, 3.0, 4.0, 5.0];
    const hist = computeHistogram(values);
    assert.strictEqual(hist.edges.length, hist.counts.length + 1);
});

test('normal case: edges[0] === min and edges[last] === max', () => {
    const values = [1.0, 2.0, 3.0, 4.0, 5.0];
    const hist = computeHistogram(values);
    assert.strictEqual(hist.edges[0], hist.min);
    assert.strictEqual(hist.edges[hist.edges.length - 1], hist.max);
});

test('uniform distribution: all values equal — all count in first bin', () => {
    const values = [3.14, 3.14, 3.14, 3.14];
    const hist = computeHistogram(values);
    assert.ok(hist !== null, 'histogram should not be null');
    assert.strictEqual(hist.counts[0], values.length);
    // All other bins should be zero
    for (let i = 1; i < hist.counts.length; i++) {
        assert.strictEqual(hist.counts[i], 0);
    }
});

test('nested array input: [[1,2],[3,4]] is handled correctly', () => {
    const values = [[1, 2], [3, 4]];
    const hist = computeHistogram(values);
    assert.ok(hist !== null, 'histogram should not be null');
    assert.strictEqual(hist.total, 4);
    assert.strictEqual(hist.min, 1);
    assert.strictEqual(hist.max, 4);
    assert.strictEqual(sumCounts(hist.counts), 4);
});

test('nested array: deeply nested input', () => {
    const values = [[[1, 2]], [[3, [4, 5]]]];
    const hist = computeHistogram(values);
    assert.ok(hist !== null);
    assert.strictEqual(hist.total, 5);
});

test('NaN values are filtered out', () => {
    const values = [1.0, NaN, 2.0, NaN, 3.0];
    const hist = computeHistogram(values);
    assert.ok(hist !== null);
    assert.strictEqual(hist.total, 3);
    assert.strictEqual(sumCounts(hist.counts), 3);
});

test('Infinity values are filtered out', () => {
    const values = [1.0, Infinity, 2.0, -Infinity, 3.0];
    const hist = computeHistogram(values);
    assert.ok(hist !== null);
    assert.strictEqual(hist.total, 3);
    assert.strictEqual(sumCounts(hist.counts), 3);
});

test('all values are NaN — returns null', () => {
    const values = [NaN, NaN, NaN];
    const hist = computeHistogram(values);
    assert.strictEqual(hist, null);
});

test('empty array returns null', () => {
    const hist = computeHistogram([]);
    assert.strictEqual(hist, null);
});

test('null input returns null', () => {
    const hist = computeHistogram(null);
    assert.strictEqual(hist, null);
});

test('undefined input returns null', () => {
    const hist = computeHistogram(undefined);
    assert.strictEqual(hist, null);
});

test('single element array', () => {
    const values = [42.0];
    const hist = computeHistogram(values);
    assert.ok(hist !== null);
    assert.strictEqual(hist.total, 1);
    assert.strictEqual(hist.min, 42.0);
    assert.strictEqual(hist.max, 42.0);
    assert.strictEqual(hist.mean, 42.0);
    assert.strictEqual(hist.std, 0);
    // range === 0 so all count goes into first bin
    assert.strictEqual(hist.counts[0], 1);
});

test('std deviation is correct for known input', () => {
    // values: [2, 4, 4, 4, 5, 5, 7, 9] — Wikipedia example, std = 2
    const values = [2, 4, 4, 4, 5, 5, 7, 9];
    const hist = computeHistogram(values);
    assert.ok(hist !== null);
    const expectedMean = 5;
    assert.ok(Math.abs(hist.mean - expectedMean) < 1e-10);
    assert.ok(Math.abs(hist.std - 2) < 1e-10, `std ${hist.std} should be 2`);
});

test('counts are non-negative integers', () => {
    const values = Array.from({ length: 100 }, (_, i) => i * 0.1);
    const hist = computeHistogram(values);
    assert.ok(hist !== null);
    for (const count of hist.counts) {
        assert.ok(count >= 0, 'count must be non-negative');
        assert.strictEqual(count, Math.floor(count), 'count must be integer');
    }
});

test('edges are monotonically increasing', () => {
    const values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    const hist = computeHistogram(values);
    assert.ok(hist !== null);
    for (let i = 1; i < hist.edges.length; i++) {
        assert.ok(hist.edges[i] >= hist.edges[i - 1], `edges must be monotonically increasing at index ${i}`);
    }
});
