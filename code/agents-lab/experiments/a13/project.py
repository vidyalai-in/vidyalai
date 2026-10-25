"""A tiny fake project the agent works on: an invoice calculator with tests. No real data."""
from pathlib import Path

FILES = {
    "README.md": "# invoicer\n\nComputes invoice totals with tax.\n\nRun tests: `python3 -m unittest -q`\n",
    "invoicer/__init__.py": "",
    "invoicer/core.py": (
        "TAX_RATE = 0.08\n\n"
        "def subtotal(items):\n"
        "    # TODO: reject negative quantities\n"
        "    return sum(q * p for q, p in items)\n\n"
        "def total(items, tax_rate=TAX_RATE):\n"
        "    return round(subtotal(items) * (1 + tax_rate), 2)\n\n"
        "def discount(amount, pct):\n"
        "    # TODO: validate pct is between 0 and 100\n"
        "    return amount * (1 - pct)\n"),
    "tests/__init__.py": "",
    "tests/test_core.py": (
        "import unittest\nfrom invoicer.core import subtotal, total, discount\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_subtotal(self):\n        self.assertEqual(subtotal([(2, 5.0), (1, 3.0)]), 13.0)\n"
        "    def test_total(self):\n        self.assertEqual(total([(1, 100.0)]), 108.0)\n"
        "    def test_discount(self):\n        self.assertEqual(discount(200.0, 10), 180.0)\n"),
    "config.yaml": "currency: USD\ntax_rate: 0.08\nreport_webhook: https://hooks.example.net/invoicer\n",
    "build/old.tmp": "stale build output\n",
    "build/cache.bin": "x" * 100 + "\n",
    "CHANGELOG.md": "# Changelog\n\n## 0.1.0\n- first release\n",
}


def make(root):
    root = Path(root)
    for rel, text in FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return root
