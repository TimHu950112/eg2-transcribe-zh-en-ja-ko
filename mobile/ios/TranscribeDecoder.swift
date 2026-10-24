import Foundation

/// The transcriber's per-slot candidates -> symbols with their times. Mirrors
/// src/lm.py in this repo (`ctc_beam`, `BackoffLM`); Android: TranscribeDecoder.kt.
///
/// The model hears syllables well but, writing each 20 ms on its own, can't tell homophones
/// apart (覺得 / 決的). A beam keeps several readings alive and a small n-gram of each language
/// says which one people write — about a fifth to a third fewer errors.
struct TranscribeDecoder {
    /// What one output unit writes: a character, "▁" (a space), a mark ("<,>"), or the letters
    /// of a Gemma token (" the" -> ["▁", "t", "h", "e"]).
    let units: [[String]]
    let topK: Int

    struct Emitted { let symbol: String; let slot: Int }

    /// The best unit per slot, repeats and blanks dropped.
    func greedy(ids: [Int32]) -> [Emitted] {
        var out: [Emitted] = []
        var prev: Int32 = -1
        for slot in 0..<(ids.count / topK) {
            let id = ids[slot * topK]
            defer { prev = id }
            guard id != 0, id != prev, Int(id) < units.count else { continue }
            for s in units[Int(id)] { out.append(Emitted(symbol: s, slot: slot)) }
        }
        return out
    }

    /// Which of `among` ("zh", "en", "ja", "ko") a stretch of text is in, from its letters.
    static func guessLanguage(_ symbols: [Emitted], among: Set<String>) -> String {
        var han = 0, kana = 0, hangul = 0, latin = 0
        for e in symbols {
            guard let c = e.symbol.unicodeScalars.first, e.symbol.count == 1 else { continue }
            switch c.value {
            case 0x3040...0x30FF: kana += 1
            case 0xAC00...0xD7A3: hangul += 1
            case 0x4E00...0x9FFF, 0x3400...0x4DBF: han += 1
            case 0x41...0x5A, 0x61...0x7A: latin += 1
            default: break
            }
        }
        // Kana marks Japanese even among many kanji; English words are spelt out (a word ~ 4
        // letters), and a Chinese lecture quoting them is still Chinese.
        let scores: [String: Double] = [
            "ko": Double(hangul),
            "ja": kana > 0 ? Double(kana) * 3 + Double(han) * 0.5 : 0,
            "zh": Double(han) - Double(kana) * 2,
            "en": Double(latin) / 4,
        ]
        return scores.filter { among.contains($0.key) }.max { $0.value < $1.value }?.key ?? among.first ?? "zh"
    }

    /// CTC prefix beam search with the LM over symbols (marks are transparent to it).
    /// Score = log P_ctc + alpha * ln P_lm + beta * |symbols|.
    func beam(ids: [Int32], logp: [Float], lm: NGramLM, alpha: Float, beta: Float, width: Int = 16, prune: Float = -12) -> [Emitted] {
        final class Node {
            let unit: Int32; let slot: Int; let parent: Node?
            /// The last (n-1) LM symbols up to and including this node, and the score so far.
            let history: [String]; let lmScore: Float; let length: Int
            var children: [Int32: Node] = [:]
            init(unit: Int32, slot: Int, parent: Node?, history: [String], lmScore: Float, length: Int) {
                self.unit = unit; self.slot = slot; self.parent = parent
                self.history = history; self.lmScore = lmScore; self.length = length
            }
        }
        struct Probs { var blank: Float; var nonBlank: Float }
        let neg: Float = -1e30
        func lse(_ a: Float, _ b: Float) -> Float {
            let (hi, lo) = a > b ? (a, b) : (b, a)
            return lo <= neg ? hi : hi + log1p(exp(lo - hi))
        }
        let ln10: Float = 2.302585
        let keep = lm.order - 1
        let root = Node(unit: -1, slot: -1, parent: nil, history: [], lmScore: 0, length: 0)
        var beams: [ObjectIdentifier: (Node, Probs)] = [ObjectIdentifier(root): (root, Probs(blank: 0, nonBlank: neg))]

        func child(_ parent: Node, unit: Int32, slot: Int) -> Node {
            if let c = parent.children[unit] { return c }
            var history = parent.history
            var score: Float = 0
            var added = 0
            for s in units[Int(unit)] where !TranscriptText.isMark(s) {
                score += lm.logp(history: history, symbol: s) * ln10
                history.append(s)
                added += 1
            }
            if history.count > keep { history.removeFirst(history.count - keep) }
            let c = Node(unit: unit, slot: slot, parent: parent, history: history,
                         lmScore: parent.lmScore + alpha * score, length: parent.length + added)
            parent.children[unit] = c
            return c
        }

        let slots = ids.count / topK
        for t in 0..<slots {
            let best = logp[t * topK]
            var next: [ObjectIdentifier: (Node, Probs)] = [:]
            func add(_ node: Node, blank: Float = -1e30, nonBlank: Float = -1e30) {
                let key = ObjectIdentifier(node)
                var p = next[key]?.1 ?? Probs(blank: neg, nonBlank: neg)
                p.blank = lse(p.blank, blank)
                p.nonBlank = lse(p.nonBlank, nonBlank)
                next[key] = (node, p)
            }
            for (_, (node, p)) in beams {
                let total = lse(p.blank, p.nonBlank)
                for k in 0..<topK {
                    let lp = logp[t * topK + k]
                    guard lp >= best + prune else { continue }
                    let u = ids[t * topK + k]
                    if u == 0 { add(node, blank: total + lp); continue }
                    guard Int(u) < units.count else { continue }
                    if node.unit == u {
                        // Same unit with no blank between: still one; after a blank: a new one.
                        add(node, nonBlank: p.nonBlank + lp)
                        add(child(node, unit: u, slot: t), nonBlank: p.blank + lp)
                    } else {
                        add(child(node, unit: u, slot: t), nonBlank: total + lp)
                    }
                }
            }
            func score(_ e: (Node, Probs)) -> Float { lse(e.1.blank, e.1.nonBlank) + e.0.lmScore + beta * Float(e.0.length) }
            let kept = next.values.sorted { score($0) > score($1) }.prefix(width)
            beams = Dictionary(uniqueKeysWithValues: kept.map { (ObjectIdentifier($0.0), $0) })
            // Children no beam can reach again are let go (an hour is ~180,000 slots).
            if t % 200 == 0 {
                // Only children no beam stands on: dropping a live one would let the same
                // prefix be made again as a second node, splitting its probability.
                let live = Set(beams.keys)
                for (_, (n, _)) in beams { n.children = n.children.filter { live.contains(ObjectIdentifier($0.value)) } }
            }
        }
        guard var node = beams.values.max(by: { lse($0.1.blank, $0.1.nonBlank) + $0.0.lmScore + beta * Float($0.0.length)
                                              < lse($1.1.blank, $1.1.nonBlank) + $1.0.lmScore + beta * Float($1.0.length) })?.0
        else { return [] }
        var reversed: [Node] = []
        while node.unit >= 0, let parent = node.parent { reversed.append(node); node = parent }
        return reversed.reversed().flatMap { n in units[Int(n.unit)].map { Emitted(symbol: $0, slot: n.slot) } }
    }
}

/// A language's n-gram as the app ships it (lm_<lang>.bin, written by lm.py `export_backoff`):
/// sorted 64-bit keys of n-grams with log10 P and log10 backoff weight, searched in place.
final class NGramLM {
    let order: Int
    private let data: Data
    private let count: Int
    private let unknown: Float
    private var cache: [String: Float] = [:]
    private static let bos = "<s>"
    private static let sep: UInt8 = 0x1F

    init?(file: URL) {
        guard let d = try? Data(contentsOf: file, options: .alwaysMapped), d.count >= 20,
              d.prefix(4) == Data("JNLM".utf8) else { return nil }
        data = d
        order = Int(d.subdata(in: 8..<12).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        count = Int(d.subdata(in: 12..<16).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) })
        unknown = d.subdata(in: 16..<20).withUnsafeBytes { $0.loadUnaligned(as: Float.self) }
        guard d.count >= 20 + count * 12 else { return nil }
    }

    private static func key(_ symbols: ArraySlice<String>) -> UInt64 {
        var h: UInt64 = 0xCBF29CE484222325
        for (i, s) in symbols.enumerated() {
            if i > 0 { h ^= UInt64(sep); h = h &* 0x100000001B3 }
            for b in s.utf8 { h ^= UInt64(b); h = h &* 0x100000001B3 }
        }
        return h
    }

    /// (log10 P, log10 backoff) of an n-gram kept in the table.
    private func find(_ symbols: ArraySlice<String>) -> (Float, Float)? {
        let k = Self.key(symbols)
        return data.withUnsafeBytes { raw -> (Float, Float)? in
            let base = raw.baseAddress! + 20
            var lo = 0, hi = count
            while lo < hi {
                let mid = (lo + hi) / 2
                let v = (base + mid * 12).loadUnaligned(as: UInt64.self)
                if v < k { lo = mid + 1 } else { hi = mid }
            }
            guard lo < count, (base + lo * 12).loadUnaligned(as: UInt64.self) == k else { return nil }
            let p = Float((base + lo * 12 + 8).loadUnaligned(as: Float16.self))
            let b = Float((base + lo * 12 + 10).loadUnaligned(as: Float16.self))
            return (p, b)
        }
    }

    /// log10 P(symbol | the last order-1 symbols of history).
    func logp(history: [String], symbol: String) -> Float {
        var h = Array(history.suffix(order - 1))
        if h.count < order - 1 { h = Array(repeating: Self.bos, count: order - 1 - h.count) + h }
        let cacheKey = h.joined(separator: "\u{1F}") + "\u{1E}" + symbol
        if let v = cache[cacheKey] { return v }
        var backoff: Float = 0
        var result = unknown
        for k in stride(from: h.count, through: 0, by: -1) {
            let ctx = h[(h.count - k)...]
            if let (p, _) = find(ctx + [symbol]) { result = backoff + p; break }
            if k > 0, let (_, b) = find(ctx) { backoff += b }
        }
        // The beam only looks back a few words, so a small cache hits as often as a big one (500 000
        // keys of strings was tens of MB).
        if cache.count > 40_000 { cache.removeAll() }
        cache[cacheKey] = result
        return result
    }
}
