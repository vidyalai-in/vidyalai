import tomlkit
doc = tomlkit.parse('a = [\n      1 # one\n     ,2\n    ]\n')
doc['a'].append(99); s = doc.as_string(); print(s)
try: print(tomlkit.parse(s)['a'])
except Exception as e: print(type(e).__name__, e)