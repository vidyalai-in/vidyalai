ISSUES = {
"sqlparse-ctas-as": """Lowercase `as` in CREATE TABLE ... AS SELECT breaks function grouping

With an uppercase statement the function call in the SELECT list is grouped as a `Function` and the alias is found:

    >>> p = sqlparse.parse('CREATE TABLE tbl1 AS SELECT coalesce(t1.col1, 0) AS col1 FROM t1')[0]
    >>> p.tokens[10].get_alias()
    'col1'

The same statement written in lowercase loses it: `coalesce(...)` is not grouped as a Function and the alias comes back as None.

    >>> p = sqlparse.parse('create table tbl1 as select coalesce(t1.col1, 0) as col1 from t1')[0]
    >>> p.tokens[10].get_alias()   # None

SQL keywords are case-insensitive, so both should parse the same way.""",

"sqlparse-real-name": """get_real_name() returns the wrong part for names with more than two dots

For a fully qualified name such as `db.schema.tbl.col`, `get_real_name()` and `get_name()` return `schema` instead of `col`:

    >>> ident = sqlparse.parse("db.schema.tbl.col")[0].tokens[0]
    >>> ident.get_real_name(), ident.get_name(), ident.get_parent_name()
    ('schema', 'schema', 'db')

The real name should be the last component (`col`); `x.y.z AS w` should likewise give real name `z` and alias `w`. Two-part names like `a.b` work fine.""",

"sqlparse-strip-ws": """strip_whitespace leaves trailing whitespace inside parentheses

    >>> sqlparse.format('( where foo = bar  ) from', strip_whitespace=True)
    '(where foo = bar ) from'

Expected `'(where foo = bar) from'`: the space before the closing parenthesis should be removed, just as the one after the opening parenthesis is.""",

"sqlparse-typed-literal": """Comparison with a typed literal (DATE '...') is not grouped

`foo = 'bar'` parses into a single `Comparison`, but a typed literal on the right-hand side does not:

    >>> p = sqlparse.parse("foo = DATE 'bar.baz'")[0]
    >>> [type(t).__name__ for t in p.tokens]
    ['Identifier', 'Token', 'Token', 'Token', 'TypedLiteral']

Expected one `Comparison` whose `left` is `foo` and whose `right` is the `TypedLiteral` `DATE 'bar.baz'`. Same for `x > DATE '2020-01-01'` and `x > TIMESTAMP '2020-01-01 00:00:00'`.""",

"sqlparse-deepcopy": """copy.deepcopy() of a parsed statement fails

    >>> import copy, sqlparse
    >>> p = sqlparse.parse('select * from foo')[0]
    >>> copy.deepcopy(p)
    TypeError: '_TokenType' object is not callable

Deep-copying a parsed statement should work and give an equal statement (`str(copy) == str(p)`).""",

"parse-month-case": """Lowercase month names crash date parsing

Matching is case-insensitive, so `{:tg}` happily matches a lowercase month name, but the conversion then fails:

    >>> parse.parse('{:tg}', '1/Jul/2011')     # works
    >>> parse.parse('{:tg}', '1/jul/2011')
    KeyError: 'jul'

The same happens with other date formats that accept month names (`{:ta}`, `{:te}`, `{:th}`, `{:tc}`, `{:ts}`), e.g. `'mon jul 16 19:20 1997'` with `{:tc}` or `'1/jULy/2011 13:05'` with `{:tg}`. Month names in any case should parse.""",

"parse-tz-minutes": """Out-of-range timezone minutes are silently accepted

    >>> parse.parse('{:tt}', '12:00 +01:60')[0].utcoffset()
    datetime.timedelta(seconds=7200)
    >>> str(parse.parse('{:tt}', '12:00 -0199')[0])
    '12:00:00-02:39'

A minutes field of 60 or more is not a valid UTC offset. Parsing should raise a `ValueError` mentioning the timezone minute instead of rolling the minutes over into hours (also for `+160` and `-1:99`).""",

"parse-time-tz": """Time-only strftime formats drop the UTC offset

    >>> r = parse.parse('{:%H:%M:%S%z}', '12:34:56+0530')
    >>> r[0], r[0].utcoffset()
    (datetime.time(12, 34, 56), None)

`datetime.strptime('12:34:56+0530', '%H:%M:%S%z').timetz()` keeps the offset (`+05:30`). parse should return the same aware `time`, for offsets like `+0530`, `-0430`, `+00:00` and `+0000`.""",

"parse-meridiem": """Lowercase am/pm and lowercase `z` are mishandled in datetime parsing

Matching is case-insensitive, but the conversion is not:

    >>> parse.parse('{:tg}', '1/7/2011 1:15 PM')[0]
    datetime.datetime(2011, 7, 1, 13, 15)
    >>> parse.parse('{:tg}', '1/7/2011 1:15 pm')[0]
    datetime.datetime(2011, 7, 1, 1, 15)        # should be 13:15
    >>> parse.parse('{:tg}', '1/7/2011 12:15 am')[0]
    datetime.datetime(2011, 7, 1, 12, 15)       # should be 00:15
    >>> parse.parse('{:ti}', '1997-07-16T19:20z')
    ValueError: invalid literal for int() with base 10: ''

Lowercase `am`/`pm` should behave like uppercase, and a lowercase `z` suffix should mean UTC like `Z`.""",

"cachetools-rr-empty": """RRCache.popitem() on an empty cache raises the custom choice function's error

`RRCache` accepts a custom `choice` function. When the cache is empty, `popitem()` should raise `KeyError` (as for every other cache), but with a custom choice it leaks whatever the function raises:

    >>> c = RRCache(maxsize=2, choice=min)
    >>> c.popitem()
    ValueError: min() iterable argument is empty

A choice function should only ever be called on a non-empty sequence. The same applies to `max` or `lambda seq: next(iter(seq))`.""",

"cachetools-stacked": """Stacking two @cached decorators makes the outer wrapper use the inner cache

    >>> inner = cachetools.cached(cachetools.LRUCache(2), info=True)(f)
    >>> outer = cachetools.cached(cachetools.LRUCache(2), info=True)(inner)
    >>> outer.cache is inner.cache
    True

After `outer(0); outer(0)` I expect `outer.cache_info()` to report 1 hit and 1 miss and `inner.cache_info()` 0 hits and 1 miss, but both report the inner wrapper's statistics, and `outer.cache_clear()` clears the wrong cache. The outer wrapper's `cache`, `cache_info` and `cache_clear` should belong to the outer decorator, while `outer.__wrapped__` is still `inner`. This happens for every cache type and with or without a lock.""",

"toolz-interpose": """interpose() on an empty sequence raises StopIteration

    >>> list(toolz.interpose('a', []))
    StopIteration

Expected `[]`.""",

"toolz-accumulate": """accumulate() on an empty sequence raises RuntimeError

    >>> from operator import add
    >>> list(toolz.accumulate(add, []))
    RuntimeError: generator raised StopIteration

Expected `[]` when no initial value is given (and `[initial]` when one is).""",

"semver-next-build": """next_version() does not bump versions that only carry build metadata

    >>> Version.parse('1.2.3+build.5').next_version('patch')
    Version(major=1, minor=2, patch=3, ...)     # str: '1.2.3'
    >>> Version.parse('1.0.0+build.5').next_version('major')
    '1.0.0'

Build metadata does not affect precedence, so `1.2.3+build.5` is a final release and its next patch is `1.2.4` (and next major of `1.0.0+build.5` is `2.0.0`, next minor of `1.2.0+build.5` is `1.3.0`). Only prerelease versions should be finalized instead of bumped; `1.2.3-rc.1+build.5` with `patch` should still give `1.2.3`.""",

"semver-replace-subclass": """Version.replace() returns a Version instead of the subclass

With a subclass of `Version` (as in the docs' "Creating Subclasses from Version" example):

    class SemVerWithVPrefix(Version):
        def __str__(self):
            return "v" + super().__str__()

    v = SemVerWithVPrefix.parse("1.1.0")
    type(v.replace(prerelease="dev.0"))   # <class 'semver.version.Version'>
    str(v.replace(prerelease="dev.0"))    # '1.1.0-dev.0', expected 'v1.1.0-dev.0'

`replace()` should return an instance of the same class it was called on.""",

"marshmallow-pluck": """Pluck fails when the plucked field has an empty data_key

    class ChildSchema(Schema):
        value = fields.Integer(data_key="")

    class ParentSchema(Schema):
        child = fields.Pluck(ChildSchema, "value")

    ParentSchema().dump({"child": {"value": 2}})
    # KeyError: 'value'

An empty string is a valid `data_key`. Expected `ParentSchema().dump({"child": {"value": 2}}) == {"child": 2}` and `load({"child": 2}) == {"child": {"value": 2}}`, also with `many=True`.""",

"marshmallow-url-fragment": """URL validator rejects a fragment directly after the host

    >>> validate.URL()('http://example.org/#fragment')    # OK
    >>> validate.URL()('http://example.org#fragment')
    ValidationError: Not a valid URL.

`http://example.org#fragment` and `http://user@example.com#fragment` are valid absolute URLs (the path is empty and a fragment follows) and should pass, with `relative=False` and with relative URLs allowed.""",

"marshmallow-enum-name": """fields.Enum by name returns non-member attributes of the enum class

    class G(enum.Enum):
        male = 1
        female = 2

    fields.Enum(G).deserialize("mro")        # <built-in method mro ...>
    fields.Enum(G).deserialize("__class__")  # <class 'enum.EnumType'>

Deserializing by name should only ever return an actual member; anything else (`mro`, `__class__`, `__members__`, ...) should raise `ValidationError` with the usual "Must be one of: ..." message.""",

"marshmallow-partial-datakey": """Nested partial loading ignores the field name when the nested field has a data_key

    class AddressSchema(Schema):
        zip_code = fields.String(required=True)
        city = fields.String(required=True)

    class PersonSchema(Schema):
        address = fields.Nested(AddressSchema, data_key="homeAddress", required=True)

    PersonSchema().load({"homeAddress": {"city": "Springfield"}}, partial=("address.zip_code",))
    # ValidationError: {'homeAddress': {'zip_code': ['Missing data for required field.']}}

`partial` takes attribute names (as everywhere else), so `"address.zip_code"` should make `zip_code` optional inside the nested schema and the load should succeed. Without `partial` the error should still be reported.""",

"mistune-empty-ref": """Markdown renderer drops the `<>` of an empty reference destination

Round-tripping a reference definition with an empty destination through `MarkdownRenderer` loses the angle brackets:

    md = mistune.create_markdown(renderer=MarkdownRenderer())
    md('[r]: <>\\n')            # '[r]: \\n'   expected '[r]: <>\\n'
    md('[r]: <> "title"\\n')    # expected '[r]: <> "title"\\n'

`[r]: ` with nothing after it is not a valid reference definition, so the output no longer means the same thing as the input.""",

"mistune-quote-gt": """Markdown renderer strips a trailing `>` from block quote content

    md = mistune.create_markdown(renderer=MarkdownRenderer())
    md('> <https://example.com>\\n')    # '> <https://example.com\\n'

The closing `>` of the autolink is lost. Same with inline HTML (`> <b>raw</b>`), a literal `>` at the end of a quoted line, nested quotes and multi-paragraph quotes ending in an autolink. Round-tripping should return the input unchanged.""",

"mistune-deflist-indent": """def_list: indented definition markers are not recognised

    md = mistune.create_markdown(plugins=['def_list'])
    md('Term\\n: definition\\n')     # <dl><dt>Term</dt><dd>definition</dd></dl>
    md('Term\\n  : definition\\n')   # '<p>Term\\n: definition</p>\\n'

Like other block markers, the `:` of a definition may be indented by up to three spaces (with a space or a tab after it, in tight and loose lists). Continuation lines and nested blocks of an indented definition should work as for an unindented one, and successive definitions may use different indentation.""",

"humanize-years": """naturaldelta truncates years instead of rounding

    >>> humanize.naturaldelta(dt.timedelta(days=365 * 2.6))
    '2 years'
    >>> humanize.naturaldelta(dt.timedelta(days=365 * 2.9))
    '2 years'

Every other unit is rounded to the nearest value (90 minutes gives "2 hours"), so these should be "3 years".""",

"humanize-intcomma": """intcomma raises OverflowError for very large integers

    >>> humanize.intcomma(10**400 + 123)
    OverflowError: int too large to convert to float

Python ints are unbounded; `intcomma` should format them like any other int ("10,000,...,000,123"), for negative values too.""",

"humanize-fractional": """fractional() gives degenerate output when the fraction rounds to a whole number

    >>> humanize.fractional(2.9999999)
    '2 1/1'
    >>> humanize.fractional(0.9999999)
    '1/1'
    >>> humanize.fractional(0)
    '0/1'

Expected '3', '1' and '0' (and '-3' for -2.9999999). Values like 1, 2.0 and 4/3 already work ('1', '2', '1 1/3').""",

"tomlkit-slice-del": """Deleting a slice from an Array corrupts its rendered output

    >>> a = tomlkit.item([1, 2, 3])
    >>> del a[:0]          # deletes nothing
    >>> list(a), a.as_string()
    ([1, 2, 3], '[]')
    >>> a = tomlkit.item([1, 2, 3]); del a[-1:]
    >>> list(a), a.as_string()
    ([1, 2], '[]')

The list contents are right, but `as_string()` is wrong for slices with a stop of 0 or negative bounds (also `del a[:-1]`, which should render `[3]`).""",

"tomlkit-esc": """dumps() writes ESC as `\\e`, which TOML 1.0 parsers reject

    >>> s = tomlkit.dumps({'k': 'a\\x1bb'})
    >>> s
    'k = "a\\\\eb"\\n'
    >>> tomllib.loads(s)
    TOMLDecodeError: Unescaped '\\' in a string (at line 1, column 9)

`\\e` only exists in TOML 1.1. The ESC character (U+001B) should be written as `\\u001b`, in keys, basic strings and multiline strings, so the output stays readable by `tomllib` and round-trips through tomlkit.""",

"tomlkit-comma-first": """Appending to a comma-first array produces invalid TOML

    doc = tomlkit.parse('a = [\\n      1 # one\\n     ,2\\n    ]\\n')
    doc['a'].append(99)
    print(doc.as_string())

gives

    a = [
          1 # one
         ,2,
         ,99
        ]

which no longer parses (`Unexpected character: ','`). In comma-first style the new item's leading comma already separates it from `2`, so no trailing comma should be added. `insert()` in the middle has the same problem; inserting at the front should keep working.""",
}
