S = {
"sqlparse-ctas-as": """import sqlparse
from sqlparse import sql
for q in ['CREATE TABLE tbl1 AS SELECT coalesce(t1.col1, 0) AS col1 FROM t1','create table tbl1 as select coalesce(t1.col1, 0) as col1 from t1']:
    p = sqlparse.parse(q)[0]; print([type(t).__name__ for t in p.tokens if not t.is_whitespace])""",
"sqlparse-real-name": """import sqlparse
i = sqlparse.parse('db.schema.tbl.col')[0].tokens[0]; print(i.get_real_name(), i.get_name(), i.get_parent_name())""",
"sqlparse-strip-ws": """import sqlparse
print(repr(sqlparse.format('( where foo = bar  ) from', strip_whitespace=True)))""",
"sqlparse-typed-literal": """import sqlparse
p = sqlparse.parse("foo = DATE 'bar.baz'")[0]; print(len(p.tokens), [type(t).__name__ for t in p.tokens])""",
"sqlparse-deepcopy": """import sqlparse, copy
p = sqlparse.parse('select * from foo')[0]
try:
    print(str(copy.deepcopy(p)))
except BaseException as e: print(type(e).__name__, str(e)[:200])""",
"parse-month-case": """import parse
print(parse.parse('{:tg}', '1/Jul/2011')); print(parse.parse('{:tg}', '1/jul/2011'))
try: print(parse.parse('{:tg}', '1/jul/2011'))
except Exception as e: print(type(e).__name__, e)""",
"parse-tz-minutes": """import parse
for o in ['+01:30', '+01:60', '-0199']:
    try: r = parse.parse('{:tt}', '12:00 ' + o); print(o, r[0], r[0].utcoffset())
    except Exception as e: print(o, type(e).__name__, e)""",
"parse-time-tz": """import parse
r = parse.parse('{:%H:%M:%S%z}', '12:34:56+0530'); print(repr(r[0]), r[0].utcoffset())""",
"parse-meridiem": """import parse
for f, s in [('{:ti}', '1997-07-16T19:20z'), ('{:tg}', '1/7/2011 1:15 PM'), ('{:tg}', '1/7/2011 1:15 pm'), ('{:tg}', '1/7/2011 12:15 am')]:
    try: print(s, repr(parse.parse(f, s)[0]))
    except Exception as e: print(s, type(e).__name__, e)""",
"cachetools-rr-empty": """from cachetools import RRCache
c = RRCache(maxsize=2, choice=min)
try: c.popitem()
except Exception as e: print(type(e).__name__, e)""",
"cachetools-stacked": """import cachetools
inner = cachetools.cached(cachetools.LRUCache(2), info=True)(lambda x: x)
outer = cachetools.cached(cachetools.LRUCache(2), info=True)(inner)
print(outer.__wrapped__ is inner, outer.cache is inner.cache)
outer(0); outer(0); print(outer.cache_info(), inner.cache_info())""",
"toolz-interpose": """from toolz import interpose
try: print(list(interpose('a', [])))
except BaseException as e: print(type(e).__name__, e)""",
"toolz-accumulate": """from toolz import accumulate
from operator import add
try: print(list(accumulate(add, [])))
except BaseException as e: print(type(e).__name__, e)""",
"semver-next-build": """from semver import Version
for v, p in [('1.2.3+build.5','patch'), ('1.0.0+build.5','major'), ('1.2.3','patch')]: print(v, p, Version.parse(v).next_version(p))""",
"semver-replace-subclass": """from semver import Version
class V(Version):
    def __str__(self): return 'v' + super().__str__()
v = V.parse('1.1.0'); print(type(v.replace(prerelease='dev.0')).__name__, str(v.replace(prerelease='dev.0')))""",
"marshmallow-pluck": """from marshmallow import Schema, fields
class C(Schema):
    value = fields.Integer(data_key='')
class P(Schema):
    child = fields.Pluck(C, 'value')
print(P().dump({'child': {'value': 2}}))
try: print(P().load({'child': 2}))
except Exception as e: print(type(e).__name__, e)""",
"marshmallow-url-fragment": """from marshmallow import validate
for u in ['http://example.org/#fragment', 'http://example.org#fragment']:
    try: print(u, validate.URL()(u))
    except Exception as e: print(u, type(e).__name__, e)""",
"marshmallow-enum-name": """import enum
from marshmallow import fields
class G(enum.Enum):
    male = 1; female = 2
f = fields.Enum(G)
for v in ['male', 'mro', '__class__']:
    try: print(v, repr(f.deserialize(v)))
    except Exception as e: print(v, type(e).__name__, e)""",
"marshmallow-partial-datakey": """from marshmallow import Schema, fields
class A(Schema):
    zip_code = fields.String(required=True); city = fields.String(required=True)
class P(Schema):
    address = fields.Nested(A, data_key='homeAddress', required=True)
try: print(P().load({'homeAddress': {'city': 'Springfield'}}, partial=('address.zip_code',)))
except Exception as e: print(type(e).__name__, e)""",
"mistune-empty-ref": """import mistune
from mistune.renderers.markdown import MarkdownRenderer
md = mistune.create_markdown(renderer=MarkdownRenderer())
print(repr(md('[r]: <>\\n')))
print(repr(md('[a][r]\\n\\n[r]: <>\\n')))""",
"mistune-quote-gt": """import mistune
from mistune.renderers.markdown import MarkdownRenderer
md = mistune.create_markdown(renderer=MarkdownRenderer())
print(repr(md('> <https://example.com>\\n')))""",
"mistune-deflist-indent": """import mistune
md = mistune.create_markdown(plugins=['def_list'])
print(repr(md('Term\\n: definition\\n'))); print(repr(md('Term\\n  : definition\\n')))""",
"humanize-years": """import humanize, datetime as dt
for d in [365*2+200, 365*3+300, 10000]: print(d, humanize.naturaldelta(dt.timedelta(days=d)))""",
"humanize-intcomma": """import humanize
try: print(humanize.intcomma(10**400 + 123)[:40])
except Exception as e: print(type(e).__name__, e)""",
"humanize-fractional": """import humanize
for x in [2.9999999, 0.9999999, 0, 0.0, 1]: print(x, repr(humanize.fractional(x)))""",
"tomlkit-slice-del": """from tomlkit import item
a = item([1, 2, 3]); del a[:0]; print(list(a), a.as_string())
a = item([1, 2, 3]); del a[-1:]; print(list(a), a.as_string())""",
"tomlkit-esc": """import tomlkit, tomllib
s = tomlkit.dumps({'k': 'a\\x1bb'}); print(repr(s))
try: print(tomllib.loads(s))
except Exception as e: print(type(e).__name__, e)""",
"tomlkit-comma-first": """import tomlkit
doc = tomlkit.parse('a = [\\n      1 # one\\n     ,2\\n    ]\\n')
doc['a'].append(99); s = doc.as_string(); print(s)
try: print(tomlkit.parse(s)['a'])
except Exception as e: print(type(e).__name__, e)""",
}
