package com.timhu.joinnote.engine

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp
import kotlin.math.ln1p

/**
 * The transcriber's per-slot candidates -> symbols with their times. Mirrors
 * src/lm.py in this repo (`ctc_beam`, `BackoffLM`); iOS: TranscribeDecoder.swift.
 *
 * The model hears syllables well but, writing each 20 ms on its own, can't tell homophones apart
 * (覺得 / 決的). A beam keeps several readings alive and a small n-gram of each language says which
 * one people write — about a fifth to a third fewer errors.
 */
class TranscribeDecoder(
    /** What one output unit writes: a character, "▁", a mark ("<,>"), or several letters. */
    private val units: List<List<String>>,
    private val topK: Int,
) {
    data class Emitted(val symbol: String, val slot: Int)

    /** The best unit per slot, repeats and blanks dropped. */
    fun greedy(ids: IntArray): List<Emitted> {
        val out = ArrayList<Emitted>()
        var prev = -1
        for (slot in 0 until ids.size / topK) {
            val id = ids[slot * topK]
            if (id != 0 && id != prev && id < units.size) units[id].forEach { out += Emitted(it, slot) }
            prev = id
        }
        return out
    }

    private class Node(val unit: Int, val slot: Int, val parent: Node?, val history: List<String>, val lmScore: Float, val length: Int) {
        val children = HashMap<Int, Node>()
    }

    private class Probs(var blank: Float, var nonBlank: Float)

    /**
     * CTC prefix beam search with the LM over symbols (marks are transparent to it).
     * Score = log P_ctc + alpha * ln P_lm + beta * |symbols|.
     */
    fun beam(ids: IntArray, logp: FloatArray, lm: NGramLM, alpha: Float, beta: Float, width: Int = 16, prune: Float = -12f): List<Emitted> {
        val neg = -1e30f
        fun lse(a: Float, b: Float): Float {
            val hi = maxOf(a, b); val lo = minOf(a, b)
            return if (lo <= neg) hi else hi + ln1p(exp(lo - hi))
        }
        val ln10 = 2.302585f
        val keep = lm.order - 1
        fun child(parent: Node, unit: Int, slot: Int): Node = parent.children.getOrPut(unit) {
            val history = ArrayList(parent.history)
            var score = 0f
            var added = 0
            for (s in units[unit]) {
                if (TranscriptText.isMark(s)) continue
                score += lm.logp(history, s) * ln10
                history.add(s); added++
            }
            while (history.size > keep) history.removeAt(0)
            Node(unit, slot, parent, history, parent.lmScore + alpha * score, parent.length + added)
        }
        val root = Node(-1, -1, null, emptyList(), 0f, 0)
        var beams = LinkedHashMap<Node, Probs>().apply { put(root, Probs(0f, neg)) }
        fun score(n: Node, p: Probs) = lse(p.blank, p.nonBlank) + n.lmScore + beta * n.length
        val slots = ids.size / topK
        for (t in 0 until slots) {
            val best = logp[t * topK]
            val next = HashMap<Node, Probs>()
            fun add(n: Node, blank: Float = neg, nonBlank: Float = neg) {
                val p = next.getOrPut(n) { Probs(neg, neg) }
                p.blank = lse(p.blank, blank); p.nonBlank = lse(p.nonBlank, nonBlank)
            }
            for ((node, p) in beams) {
                val total = lse(p.blank, p.nonBlank)
                for (k in 0 until topK) {
                    val lp = logp[t * topK + k]
                    if (lp < best + prune) continue
                    val u = ids[t * topK + k]
                    if (u == 0) { add(node, blank = total + lp); continue }
                    if (u >= units.size) continue
                    if (node.unit == u) {
                        // Same unit with no blank between: still one; after a blank: a new one.
                        add(node, nonBlank = p.nonBlank + lp)
                        add(child(node, u, t), nonBlank = p.blank + lp)
                    } else add(child(node, u, t), nonBlank = total + lp)
                }
            }
            beams = LinkedHashMap<Node, Probs>().apply {
                next.entries.sortedByDescending { score(it.key, it.value) }.take(width).forEach { put(it.key, it.value) }
            }
            // Children no beam can reach again are let go (an hour is ~180,000 slots).
            if (t % 200 == 0) beams.keys.forEach { n -> n.children.keys.retainAll(setOf(n.unit)) }
        }
        var node = beams.entries.maxByOrNull { score(it.key, it.value) }?.key ?: return emptyList()
        val reversed = ArrayList<Node>()
        while (node.unit >= 0) { reversed += node; node = node.parent ?: break }
        return reversed.asReversed().flatMap { n -> units[n.unit].map { Emitted(it, n.slot) } }
    }

    companion object {
        /** Which of `among` ("zh", "en", "ja", "ko") a stretch of text is in, from its letters. */
        fun guessLanguage(symbols: List<Emitted>, among: Set<String>): String {
            var han = 0; var kana = 0; var hangul = 0; var latin = 0
            for (e in symbols) {
                if (e.symbol.codePointCount(0, e.symbol.length) != 1) continue
                when (e.symbol.codePointAt(0)) {
                    in 0x3040..0x30FF -> kana++
                    in 0xAC00..0xD7A3 -> hangul++
                    in 0x4E00..0x9FFF, in 0x3400..0x4DBF -> han++
                    in 0x41..0x5A, in 0x61..0x7A -> latin++
                }
            }
            // Kana marks Japanese even among many kanji; English words are spelt out (a word ~ 4
            // letters), and a Chinese lecture quoting them is still Chinese.
            val scores = mapOf(
                "ko" to hangul.toDouble(),
                "ja" to if (kana > 0) kana * 3.0 + han * 0.5 else 0.0,
                "zh" to han - kana * 2.0,
                "en" to latin / 4.0,
            )
            return scores.filterKeys { it in among }.maxByOrNull { it.value }?.key ?: among.firstOrNull() ?: "zh"
        }
    }
}

/**
 * A language's n-gram as the app ships it (lm_<lang>.bin, written by ml/.../lm_big.py): sorted
 * 64-bit keys of n-grams with log10 P and log10 backoff weight, searched in place (memory-mapped).
 */
class NGramLM private constructor(private val buf: ByteBuffer, val order: Int, private val count: Int, private val unknown: Float) {
    private val cache = HashMap<String, Float>()

    private fun key(symbols: List<String>, from: Int, to: Int, extra: String?): Long {
        var h = -0x340d631b7bdddcdbL  // 0xCBF29CE484222325
        val prime = 0x100000001B3L
        var first = true
        fun feed(s: String) {
            if (!first) { h = h xor 0x1FL; h *= prime }
            first = false
            for (b in s.toByteArray(Charsets.UTF_8)) { h = h xor (b.toLong() and 0xFF); h *= prime }
        }
        for (i in from until to) feed(symbols[i])
        if (extra != null) feed(extra)
        return h
    }

    /** (log10 P, log10 backoff) of a kept n-gram, or null. */
    private fun find(k: Long): Pair<Float, Float>? {
        var lo = 0; var hi = count
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            val v = buf.getLong(20 + mid * 12)
            if (java.lang.Long.compareUnsigned(v, k) < 0) lo = mid + 1 else hi = mid
        }
        if (lo >= count || buf.getLong(20 + lo * 12) != k) return null
        return half(buf.getShort(20 + lo * 12 + 8)) to half(buf.getShort(20 + lo * 12 + 10))
    }

    /** log10 P(symbol | the last order-1 symbols of history). */
    fun logp(history: List<String>, symbol: String): Float {
        val h = ArrayList<String>(order - 1)
        val tail = history.takeLast(order - 1)
        repeat(order - 1 - tail.size) { h += BOS }
        h += tail
        val cacheKey = h.joinToString("\u001F") + "\u001E" + symbol
        cache[cacheKey]?.let { return it }
        var backoff = 0f
        var result = unknown
        for (k in h.size downTo 0) {
            val from = h.size - k
            val hit = find(key(h, from, h.size, symbol))
            if (hit != null) { result = backoff + hit.first; break }
            if (k > 0) find(key(h, from, h.size, null))?.let { backoff += it.second }
        }
        // The beam only looks back a few words, so a small cache hits as often as a big one (500 000
        // keys of strings was tens of MB).
        if (cache.size > 40_000) cache.clear()
        cache[cacheKey] = result
        return result
    }

    companion object {
        private const val BOS = "<s>"

        fun load(file: File): NGramLM? = runCatching {
            val channel = RandomAccessFile(file, "r").channel
            val buf = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).order(ByteOrder.LITTLE_ENDIAN)
            require((0 until 4).map { buf.get(it).toInt().toChar() }.joinToString("") == "JNLM")
            val order = buf.getInt(8); val count = buf.getInt(12); val unk = buf.getFloat(16)
            require(buf.capacity() >= 20 + count * 12)
            NGramLM(buf, order, count, unk)
        }.getOrNull()

        /** IEEE half -> float. */
        private fun half(bits: Short): Float {
            val h = bits.toInt() and 0xFFFF
            val sign = (h ushr 15) and 1
            val exp = (h ushr 10) and 0x1F
            val mant = h and 0x3FF
            val f = when {
                exp == 0 -> mant / 1024f * 6.1035156e-5f
                exp == 31 -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                else -> (1 + mant / 1024f) * Math.pow(2.0, (exp - 15).toDouble()).toFloat()
            }
            return if (sign == 1) -f else f
        }
    }
}
