const NUMBER_WORDS = ['Zero', 'One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Eleven', 'Twelve'];

/** "Five" for 5, for counts in page copy. Falls back to digits past twelve. */
export const numberWord = (n: number) => NUMBER_WORDS[n] ?? String(n);
