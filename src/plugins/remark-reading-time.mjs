import getReadingTime from 'reading-time';
import { toString } from 'mdast-util-to-string';

/** Adds `minutesRead` (a number) to each Markdown entry's remark frontmatter. */
export function remarkReadingTime() {
  return (tree, file) => {
    const stats = getReadingTime(toString(tree));
    file.data.astro.frontmatter.minutesRead = Math.max(1, Math.round(stats.minutes));
  };
}
