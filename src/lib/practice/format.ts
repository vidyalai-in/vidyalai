/**
 * Inline formatting for question, option and explanation text from frontmatter:
 * **bold**, *italic*, `code` and line breaks. Everything else is escaped.
 * Underscores are left alone so blanks like "2, 4, __, 8" stay as typed.
 */
export function fmt(text: string): string {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?!\*)/g, '$1<em>$2</em>')
    .trim()
    .replace(/\n{2,}/g, '</p><p>')
    .replace(/\n/g, '<br>');
}

/** Same, wrapped in paragraphs (for explanations). */
export const fmtBlock = (text: string) => `<p>${fmt(text)}</p>`;

/**
 * True when an explanation names options by letter ("**B**", "Option C", "A shows ..."). Those questions keep
 * their option order on retakes, since shuffling would make the explanation point at the wrong option.
 * Errs towards "names letters" (e.g. code letters in coding questions), which only means fewer shuffles.
 */
export const namesOptionLetters = (explanation: string) =>
  /\*\*[A-E]\*\*|\b[Oo]ptions? [A-E]\b|(?<![\w'’-])[B-E](?![\w'’-])|(?<![\w'’-])A(?=[),.:;]| (?:and|or|shows|has|gives|keeps|puts|draws|cannot|would|only|also|fails|is (?:wrong|not|the|right|correct))\b)/.test(
    explanation,
  );
