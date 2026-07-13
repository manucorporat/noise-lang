;; SIMD probe kernel: the pi indicator — X ~ unif(-1,1), Y ~ unif(-1,1), X^2 + Y^2 < 1.
;;
;; A hand-written f64x2 twin of what `wasm_emit` emits as scalar code, to answer ONE question before
;; committing to a vector emitter: does wasm SIMD actually beat the existing 4-independent-stream
;; scalar kernel in V8?
;;
;; It is not obvious that it does. PERF.md records a hand-written NEON probe on the native side that
;; *lost* 13-16% on exactly this graph class (pi 0.87x, dice 0.84x): the scalar kernel runs its
;; integer RNG and its FP math on disjoint execution ports that the out-of-order core overlaps for
;; free, while a vector kernel makes both compete for the same vector pipes. Multi-stream scalar
;; already harvests the instruction-level parallelism SIMD would sell you. V8's baseline is weaker
;; than LLVM's, so the answer may differ here — but it must be measured, not assumed.
;;
;; Same ABI as the emitted kernels: kernel(out_ptr, n, state_ptr) writes n f64s at out_ptr and carries
;; the xoshiro state at state_ptr. FOUR lanes per iteration (two i64x2 stream-pairs), matching the
;; scalar kernel's four streams, so the comparison is like-for-like.
;;
;; One difference is forced by the ISA: the scalar kernel turns a u64 into a double with
;; `f64.convert_i64_u`, and **wasm SIMD has no i64x2 -> f64x2 convert**. So the vector path builds the
;; double by bit-surgery: take the top 52 bits, or in the exponent of 1.0 to land in [1,2), subtract
;; 1.0 to get [0,1). That is a 52-bit mantissa rather than the scalar path's 53 — distributionally
;; equivalent for Monte Carlo, but a real difference a production emitter would have to reconcile
;; against the interpreter oracle.
(module
  (memory (export "memory") 2)

  (func (export "kernel") (param $out i32) (param $n i32) (param $state i32)
    (local $i i32) (local $p i32)
    ;; two stream-pairs (A, B); each v128 holds the same xoshiro word for two independent lanes
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local $b0 v128) (local $b1 v128) (local $b2 v128) (local $b3 v128)
    (local $r v128) (local $t v128) (local $xa v128) (local $ya v128) (local $xb v128) (local $yb v128)

    (local.set $a0 (v128.load offset=0   (local.get $state)))
    (local.set $a1 (v128.load offset=16  (local.get $state)))
    (local.set $a2 (v128.load offset=32  (local.get $state)))
    (local.set $a3 (v128.load offset=48  (local.get $state)))
    (local.set $b0 (v128.load offset=64  (local.get $state)))
    (local.set $b1 (v128.load offset=80  (local.get $state)))
    (local.set $b2 (v128.load offset=96  (local.get $state)))
    (local.set $b3 (v128.load offset=112 (local.get $state)))

    (local.set $p (local.get $out))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $n)))

        ;; ================= pair A, draw X =================
        ;; r = rotl(a0 + a3, 23) + a0
        (local.set $t (i64x2.add (local.get $a0) (local.get $a3)))
        (local.set $r (i64x2.add
          (v128.or (i64x2.shl (local.get $t) (i32.const 23))
                   (i64x2.shr_u (local.get $t) (i32.const 41)))
          (local.get $a0)))
        (local.set $t (i64x2.shl (local.get $a1) (i32.const 17)))
        (local.set $a2 (v128.xor (local.get $a2) (local.get $a0)))
        (local.set $a3 (v128.xor (local.get $a3) (local.get $a1)))
        (local.set $a1 (v128.xor (local.get $a1) (local.get $a2)))
        (local.set $a0 (v128.xor (local.get $a0) (local.get $a3)))
        (local.set $a2 (v128.xor (local.get $a2) (local.get $t)))
        (local.set $a3 (v128.or (i64x2.shl (local.get $a3) (i32.const 45))
                                (i64x2.shr_u (local.get $a3) (i32.const 19))))
        ;; u in [0,1) via bit-surgery, then X = -1 + 2u
        (local.set $xa (f64x2.sub
          (f64x2.mul
            (f64x2.sub
              (v128.or (i64x2.shr_u (local.get $r) (i32.const 12))
                       (v128.const i64x2 0x3FF0000000000000 0x3FF0000000000000))
              (f64x2.splat (f64.const 1)))
            (f64x2.splat (f64.const 2)))
          (f64x2.splat (f64.const 1))))

        ;; ================= pair A, draw Y =================
        (local.set $t (i64x2.add (local.get $a0) (local.get $a3)))
        (local.set $r (i64x2.add
          (v128.or (i64x2.shl (local.get $t) (i32.const 23))
                   (i64x2.shr_u (local.get $t) (i32.const 41)))
          (local.get $a0)))
        (local.set $t (i64x2.shl (local.get $a1) (i32.const 17)))
        (local.set $a2 (v128.xor (local.get $a2) (local.get $a0)))
        (local.set $a3 (v128.xor (local.get $a3) (local.get $a1)))
        (local.set $a1 (v128.xor (local.get $a1) (local.get $a2)))
        (local.set $a0 (v128.xor (local.get $a0) (local.get $a3)))
        (local.set $a2 (v128.xor (local.get $a2) (local.get $t)))
        (local.set $a3 (v128.or (i64x2.shl (local.get $a3) (i32.const 45))
                                (i64x2.shr_u (local.get $a3) (i32.const 19))))
        (local.set $ya (f64x2.sub
          (f64x2.mul
            (f64x2.sub
              (v128.or (i64x2.shr_u (local.get $r) (i32.const 12))
                       (v128.const i64x2 0x3FF0000000000000 0x3FF0000000000000))
              (f64x2.splat (f64.const 1)))
            (f64x2.splat (f64.const 2)))
          (f64x2.splat (f64.const 1))))

        ;; ================= pair B, draw X =================
        (local.set $t (i64x2.add (local.get $b0) (local.get $b3)))
        (local.set $r (i64x2.add
          (v128.or (i64x2.shl (local.get $t) (i32.const 23))
                   (i64x2.shr_u (local.get $t) (i32.const 41)))
          (local.get $b0)))
        (local.set $t (i64x2.shl (local.get $b1) (i32.const 17)))
        (local.set $b2 (v128.xor (local.get $b2) (local.get $b0)))
        (local.set $b3 (v128.xor (local.get $b3) (local.get $b1)))
        (local.set $b1 (v128.xor (local.get $b1) (local.get $b2)))
        (local.set $b0 (v128.xor (local.get $b0) (local.get $b3)))
        (local.set $b2 (v128.xor (local.get $b2) (local.get $t)))
        (local.set $b3 (v128.or (i64x2.shl (local.get $b3) (i32.const 45))
                                (i64x2.shr_u (local.get $b3) (i32.const 19))))
        (local.set $xb (f64x2.sub
          (f64x2.mul
            (f64x2.sub
              (v128.or (i64x2.shr_u (local.get $r) (i32.const 12))
                       (v128.const i64x2 0x3FF0000000000000 0x3FF0000000000000))
              (f64x2.splat (f64.const 1)))
            (f64x2.splat (f64.const 2)))
          (f64x2.splat (f64.const 1))))

        ;; ================= pair B, draw Y =================
        (local.set $t (i64x2.add (local.get $b0) (local.get $b3)))
        (local.set $r (i64x2.add
          (v128.or (i64x2.shl (local.get $t) (i32.const 23))
                   (i64x2.shr_u (local.get $t) (i32.const 41)))
          (local.get $b0)))
        (local.set $t (i64x2.shl (local.get $b1) (i32.const 17)))
        (local.set $b2 (v128.xor (local.get $b2) (local.get $b0)))
        (local.set $b3 (v128.xor (local.get $b3) (local.get $b1)))
        (local.set $b1 (v128.xor (local.get $b1) (local.get $b2)))
        (local.set $b0 (v128.xor (local.get $b0) (local.get $b3)))
        (local.set $b2 (v128.xor (local.get $b2) (local.get $t)))
        (local.set $b3 (v128.or (i64x2.shl (local.get $b3) (i32.const 45))
                                (i64x2.shr_u (local.get $b3) (i32.const 19))))
        (local.set $yb (f64x2.sub
          (f64x2.mul
            (f64x2.sub
              (v128.or (i64x2.shr_u (local.get $r) (i32.const 12))
                       (v128.const i64x2 0x3FF0000000000000 0x3FF0000000000000))
              (f64x2.splat (f64.const 1)))
            (f64x2.splat (f64.const 2)))
          (f64x2.splat (f64.const 1))))

        ;; indicator: (X*X + Y*Y < 1) ? 1.0 : 0.0
        ;; f64x2.lt yields an all-ones / all-zeros mask per lane, so AND-ing it with the bits of 1.0
        ;; selects 1.0 or 0.0 without a branch — the vector form of the scalar kernel's select.
        (v128.store (local.get $p)
          (v128.and
            (f64x2.lt
              (f64x2.add (f64x2.mul (local.get $xa) (local.get $xa))
                         (f64x2.mul (local.get $ya) (local.get $ya)))
              (f64x2.splat (f64.const 1)))
            (f64x2.splat (f64.const 1))))
        (v128.store offset=16 (local.get $p)
          (v128.and
            (f64x2.lt
              (f64x2.add (f64x2.mul (local.get $xb) (local.get $xb))
                         (f64x2.mul (local.get $yb) (local.get $yb)))
              (f64x2.splat (f64.const 1)))
            (f64x2.splat (f64.const 1))))

        (local.set $p (i32.add (local.get $p) (i32.const 32)))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $loop)
      )
    )

    ;; carry the advanced state back out, exactly as the emitted kernels do
    (v128.store offset=0   (local.get $state) (local.get $a0))
    (v128.store offset=16  (local.get $state) (local.get $a1))
    (v128.store offset=32  (local.get $state) (local.get $a2))
    (v128.store offset=48  (local.get $state) (local.get $a3))
    (v128.store offset=64  (local.get $state) (local.get $b0))
    (v128.store offset=80  (local.get $state) (local.get $b1))
    (v128.store offset=96  (local.get $state) (local.get $b2))
    (v128.store offset=112 (local.get $state) (local.get $b3))
  )
)
