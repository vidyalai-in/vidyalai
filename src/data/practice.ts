/**
 * The unlisted /practice/ section: MCQ mental-ability sets for Class 7–8.
 * Nothing on the main site links here, and none of these pages are in the sitemap, feed or search.
 * Each topic ("family") is split into short parts of about 12 questions (10–15 minutes).
 */

export const PRACTICE_AREAS = [
  {
    slug: 'numerical',
    name: 'Numerical ability',
    blurb: 'Work with numbers quickly and accurately: factors, fractions, ratios, speed and time, patterns, data.',
  },
  {
    slug: 'verbal',
    name: 'Verbal',
    blurb: 'Words and the rules hiding in them: vocabulary, word skills, analogies, codes, reading and grammar.',
  },
  {
    slug: 'memory',
    name: 'Memory and concentration',
    blurb: 'Look carefully, hold it in your head, then answer without looking back.',
  },
  {
    slug: 'visual',
    name: 'Visual',
    blurb: 'See shapes in your mind: dice and cubes, mirror images, paper folding, patterns, hidden figures.',
  },
  {
    slug: 'analytical',
    name: 'Analytical thinking',
    blurb: 'Break a problem into clues and put them in order: relations, directions, arrangements, logic, puzzles.',
  },
  {
    slug: 'mock',
    name: 'Mock tests',
    blurb: 'Mixed questions from every area, timed. Do these after the topic sets.',
  },
] as const;

export type PracticeAreaSlug = (typeof PRACTICE_AREAS)[number]['slug'];

export const PRACTICE_FAMILIES = [
  // Numerical ability
  { slug: 'number-sense', area: 'numerical', name: 'Number sense: factors, multiples, HCF and LCM' },
  { slug: 'fractions-percentages-ratio', area: 'numerical', name: 'Fractions, percentages and ratio' },
  { slug: 'speed-time-work', area: 'numerical', name: 'Speed, time, work and ages' },
  { slug: 'number-series', area: 'numerical', name: 'Number series and missing numbers' },
  { slug: 'calendar-clock', area: 'numerical', name: 'Calendar and clock' },
  { slug: 'symbols-operations', area: 'numerical', name: 'Symbols and operations' },
  { slug: 'data-handling', area: 'numerical', name: 'Data handling: tables and charts' },
  { slug: 'integers-rationals', area: 'numerical', name: 'Integers and rational numbers' },
  { slug: 'algebra-basics', area: 'numerical', name: 'Algebra: expressions, equations and exponents' },
  { slug: 'geometry-mensuration', area: 'numerical', name: 'Geometry and mensuration' },
  // Verbal
  { slug: 'vocabulary', area: 'verbal', name: 'Vocabulary' },
  { slug: 'word-skills', area: 'verbal', name: 'Word skills' },
  { slug: 'analogies-odd-one-out', area: 'verbal', name: 'Analogies and odd one out' },
  { slug: 'coding-decoding', area: 'verbal', name: 'Coding and decoding' },
  { slug: 'reading-grammar', area: 'verbal', name: 'Reading and grammar' },
  // Memory and concentration
  { slug: 'picture-memory', area: 'memory', name: 'Picture memory' },
  { slug: 'number-word-memory', area: 'memory', name: 'Number and word memory' },
  { slug: 'attention-detail', area: 'memory', name: 'Attention to detail' },
  // Visual
  { slug: 'cubes-dice', area: 'visual', name: 'Cubes and dice' },
  { slug: 'mirror-water-images', area: 'visual', name: 'Mirror and water images' },
  { slug: 'paper-folding', area: 'visual', name: 'Paper folding and cutting' },
  { slug: 'figure-patterns', area: 'visual', name: 'Figure series, analogies and completion' },
  { slug: 'figure-counting', area: 'visual', name: 'Counting figures and hidden shapes' },
  // Analytical thinking
  { slug: 'blood-relations', area: 'analytical', name: 'Blood relations' },
  { slug: 'direction-sense', area: 'analytical', name: 'Direction sense' },
  { slug: 'ranking-order', area: 'analytical', name: 'Ranking and order' },
  { slug: 'syllogisms-venn', area: 'analytical', name: 'Syllogisms and Venn diagrams' },
  { slug: 'logic-statements', area: 'analytical', name: 'If-then, truth-tellers and conclusions' },
  { slug: 'seating-scheduling', area: 'analytical', name: 'Seating and scheduling puzzles' },
  { slug: 'counting-pigeonhole', area: 'analytical', name: 'Counting and the pigeonhole idea' },
  { slug: 'number-puzzles', area: 'analytical', name: 'Number puzzles: magic squares, grids and codes' },
  // Mocks
  { slug: 'mock-tests', area: 'mock', name: 'Mock tests' },
] as const satisfies readonly { slug: string; area: PracticeAreaSlug; name: string }[];

export type PracticeFamilySlug = (typeof PRACTICE_FAMILIES)[number]['slug'];
export const PRACTICE_FAMILY_SLUGS = PRACTICE_FAMILIES.map((f) => f.slug) as [PracticeFamilySlug, ...PracticeFamilySlug[]];
export const practiceFamily = (slug: string) => PRACTICE_FAMILIES.find((f) => f.slug === slug)!;
export const practiceArea = (slug: string) => PRACTICE_AREAS.find((a) => a.slug === slug)!;
export const practiceUrl = (slug: string) => `/practice/${slug}/`;
