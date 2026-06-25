// The example gallery. Code is pulled live from the repo's top-level `examples/*.noise` files
// (so the site never drifts from what the CLI runs); the curated metadata below adds a title,
// a one-line blurb, and a plain-language "what's going on" explanation for each.

const rawFiles = import.meta.glob('../../../examples/*.noise', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const codeByName: Record<string, string> = {};
for (const [path, src] of Object.entries(rawFiles)) {
  const name = path.split('/').pop()!.replace(/\.noise$/, '');
  codeByName[name] = (src as string).trimEnd() + '\n';
}

export interface Example {
  id: string;
  title: string;
  /** One-line summary shown on the gallery card. */
  blurb: string;
  /** A short prose walkthrough of what the program does and why the answer is what it is. */
  explanation: string;
  /** The closed-form / analytic answer, for "you can check this" callouts. */
  analytic?: string;
  /** Grouping bucket (keeps the list scalable as examples are added). */
  category: string;
  tags: string[];
  code: string;
}

/** Display order of the gallery / dropdown sections. */
export const categories = [
  'Basics',
  'Probability',
  'Games & risk',
  'Continuous & CLT',
  'Signals & DSP',
  'Functions & research',
] as const;

/** Which bucket each example belongs to. */
const categoryOf: Record<string, (typeof categories)[number]> = {
  pi: 'Basics',
  dice: 'Basics',
  dice_sum: 'Basics',
  coin_streak: 'Probability',
  exactly_two_heads: 'Probability',
  birthday: 'Probability',
  monty_hall: 'Probability',
  conditional_bayes: 'Probability',
  advantage: 'Games & risk',
  max_of_dice: 'Games & risk',
  dice_bet: 'Games & risk',
  insurance: 'Games & risk',
  reliability: 'Games & risk',
  irwin_hall: 'Continuous & CLT',
  clt_normal: 'Continuous & CLT',
  am_vs_fm: 'Signals & DSP',
  nyquist: 'Signals & DSP',
  functions: 'Functions & research',
  qjl_scalar: 'Functions & research',
  turboquant: 'Functions & research',
};

/** Curated metadata, in pedagogical order. Code + category are injected below. */
const meta: Omit<Example, 'code' | 'category'>[] = [
  {
    id: 'pi',
    title: 'Estimate π',
    blurb: 'Monte Carlo π from random darts in a square.',
    explanation:
      'Throw darts uniformly at the 2×2 square around the origin. A dart lands inside the unit circle when X²+Y² < 1. The fraction that do is the ratio of areas, π/4 — so 4·P(inside) ≈ π. This is the "hello world" of Monte Carlo: a geometric probability turned into a number by sampling.',
    analytic: '3.14159…',
    tags: ['basics', 'monte carlo'],
  },
  {
    id: 'dice',
    title: 'A fair die',
    blurb: 'Why dice need unif_int, not unif.',
    explanation:
      'A die is discrete, so it uses unif_int(1,6) — integers 1..6. With the continuous unif(1,6) the probability of landing exactly on 4 would be 0 (a continuous draw never hits a single point). P(Dice == 4) ≈ 1/6 is the payoff of picking the right distribution.',
    analytic: '1/6 ≈ 0.1667',
    tags: ['basics', 'discrete'],
  },
  {
    id: 'dice_sum',
    title: 'Two dice',
    blurb: 'Independence is a separate ~ draw, not a repeated name.',
    explanation:
      'The one rule that surprises everyone: a name bound with ~ is one fixed draw, so Dice + Dice would be 2·Dice (one die doubled), not two dice. Real independence comes from separate draws — here iid(unif_int(1,6), 2) makes two independent dice, and sum(...) adds them. P(sum == 7) ≈ 1/6.',
    analytic: '1/6 ≈ 0.1667',
    tags: ['independence', 'collections'],
  },
  {
    id: 'coin_streak',
    title: 'Three heads in a row',
    blurb: 'Model the event, don’t multiply probabilities by hand.',
    explanation:
      'Instead of computing (1/2)³ yourself, you model three independent coin flips and ask whether all three came up heads. The language does the probability — P(all heads) ≈ 0.125 — by simulating the actual event.',
    analytic: '1/8 = 0.125',
    tags: ['independence', 'modeling'],
  },
  {
    id: 'exactly_two_heads',
    title: 'Exactly two heads',
    blurb: 'A tiny Binomial built from boolean events.',
    explanation:
      'Flip three coins and count the heads with count([...]). Asking P(count == 2) gives the Binomial(3, ½) probability of exactly two heads — built from primitives, not a formula.',
    analytic: '3/8 = 0.375',
    tags: ['modeling', 'collections'],
  },
  {
    id: 'birthday',
    title: 'Birthday paradox',
    blurb: 'How often does a group share a birthday?',
    explanation:
      'Draw a birthday for each person with iid(unif_int(1,365), n), then has_duplicate(...) checks whether any two match. Even small groups collide more often than intuition suggests — and the whole experiment is one expression that scales to any group size.',
    tags: ['collections', 'classic'],
  },
  {
    id: 'monty_hall',
    title: 'Monty Hall',
    blurb: 'Switching doors wins 2/3 of the time.',
    explanation:
      'Reframed cleanly: switching wins exactly when your first pick was wrong, which happens 2/3 of the time. Sampling the first pick and checking that condition reproduces the famous counterintuitive answer.',
    analytic: '2/3 ≈ 0.6667',
    tags: ['classic', 'modeling'],
  },
  {
    id: 'conditional_bayes',
    title: 'Conditional probability',
    blurb: 'P(A | B) as a ratio of probabilities.',
    explanation:
      'Without a built-in conditioning operator yet, a conditional probability is computed directly as P(A and B) / P(B). Here: given a die rolled above 3, how likely was it a 6? The ratio gives 1/3.',
    analytic: '1/3 ≈ 0.3333',
    tags: ['probability', 'bayes'],
  },
  {
    id: 'advantage',
    title: 'D&D advantage',
    blurb: 'Keep the higher of two d20s.',
    explanation:
      'Rolling with "advantage" means taking the max of two twenty-sided dice. A lifted if (if A > B { A } else { B }) builds the max as a new random variable, then P(result ≥ 15) shows how much advantage shifts the odds.',
    tags: ['games', 'lifted if'],
  },
  {
    id: 'max_of_dice',
    title: 'Max of two dice',
    blurb: 'max over random variables via a lifted if.',
    explanation:
      'if cond { a } else { b } over a random condition is not control flow — it is a per-sample select that builds a new random variable. That gives you max/min/abs over distributions for free. Here P(higher of 2d6 == 6) ≈ 11/36.',
    analytic: '11/36 ≈ 0.3056',
    tags: ['lifted if'],
  },
  {
    id: 'dice_bet',
    title: 'A dice bet',
    blurb: 'Build a payoff distribution, then ask about profit.',
    explanation:
      'A wager turns a die roll into a payoff random variable with if (win/lose amounts), and P(profit > 0) reports how often you come out ahead. The same lifted-if machinery models real decisions.',
    tags: ['lifted if', 'risk'],
  },
  {
    id: 'insurance',
    title: 'Insurance payout',
    blurb: 'A deductible as a lifted if over a loss.',
    explanation:
      'Model a random loss, apply a deductible with an if, and ask how often the insurer actually pays. A compact template for any threshold/payout problem.',
    tags: ['risk', 'lifted if'],
  },
  {
    id: 'reliability',
    title: 'Redundancy',
    blurb: 'Three parallel components, each 90% reliable.',
    explanation:
      'Three independent components are each up with probability 0.9; the system is up if any of them is. any([...]) over the three Bernoulli draws gives P(system up) ≈ 0.999 — redundancy turning 90% into three nines.',
    analytic: '0.999',
    tags: ['engineering', 'collections'],
  },
  {
    id: 'irwin_hall',
    title: 'Sum of uniforms',
    blurb: 'The Irwin–Hall distribution by simulation.',
    explanation:
      'Add three independent U(0,1) draws and ask P(sum > 2). The sum has a bell-ish (Irwin–Hall) shape; sampling recovers the tail probability without any density algebra.',
    analytic: '1/6 ≈ 0.1667',
    tags: ['continuous', 'collections'],
  },
  {
    id: 'clt_normal',
    title: 'Central limit theorem',
    blurb: 'A normal built from twelve uniforms.',
    explanation:
      'Summing twelve U(0,1) draws and centering gives an approximately standard-normal variable (a classic CLT trick). A tail probability of that sum lands near the true normal tail — the CLT made tangible.',
    tags: ['continuous', 'clt'],
  },
  {
    id: 'functions',
    title: 'User functions',
    blurb: 'Deterministic (=) vs stochastic (~) functions.',
    explanation:
      'f(a,b) = … is a pure function that lifts over random variables (max here). roll() ~ unif_int(1,6) is stochastic: every call draws fresh, so roll() + roll() is genuinely two dice. The = / ~ split applies to functions just like bindings.',
    tags: ['functions', 'language'],
  },
  {
    id: 'qjl_scalar',
    title: 'QJL in 1-D',
    blurb: 'An unbiased 1-bit quantizer (TurboQuant building block).',
    explanation:
      'A scalar warm-up for the TurboQuant capstone: a sign-based 1-bit sketch, rescaled by √(π/2), recovers an inner product without bias. Uses normal, E/Var, sqrt and pi — the primitives the d-dimensional version scales up.',
    tags: ['research', 'continuous'],
  },
  {
    id: 'turboquant',
    title: 'TurboQuant',
    blurb: 'Reproducing a quantization bias (and its fix) from an arXiv paper.',
    explanation:
      'The capstone. A fresh d×d Gaussian projection per sample, matrix–vector products, and reductions show that a naive MSE 1-bit quantizer is inner-product biased by 2/π and carries ~3× the squared error — while the QJL rescaling is unbiased. Empirical validation of a research result in ~20 readable lines.',
    analytic: 'bias 2/π ≈ 0.637',
    tags: ['research', 'capstone'],
  },
  {
    id: 'am_vs_fm',
    title: 'AM vs FM',
    blurb: 'Why FM survives noise better than AM.',
    explanation:
      'A full modulate → add static → demodulate pipeline for both AM (message in the amplitude) and FM (message in the angle). Given the same noise, FM recovers the signal several times cleaner. Uses lazy signals, the sin/cos/atan ufuncs, and array broadcasting.',
    tags: ['signals', 'dsp'],
  },
  {
    id: 'nyquist',
    title: 'Nyquist–Shannon',
    blurb: 'Aliasing by counterexample.',
    explanation:
      'Sample a 7-cycle wave below twice its frequency and it becomes indistinguishable from a 3-cycle wave (identical samples — aliasing). Sample above the Nyquist rate and they separate. The sampling theorem, demonstrated rather than asserted.',
    tags: ['signals', 'dsp'],
  },
];

export const examples: Example[] = meta
  .filter((m) => codeByName[m.id])
  .map((m) => ({ ...m, category: categoryOf[m.id] ?? 'Basics', code: codeByName[m.id] }));

export const defaultExampleId = 'pi';

/** Examples grouped by category, in `categories` order (empty groups dropped). */
export function examplesByCategory(): { category: string; items: Example[] }[] {
  return categories
    .map((category) => ({ category, items: examples.filter((e) => e.category === category) }))
    .filter((g) => g.items.length > 0);
}
