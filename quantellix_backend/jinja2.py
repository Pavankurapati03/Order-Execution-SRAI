from jinja2 import Environment

def environment(**options):
    env = Environment(**options)
    def intcomma(val):
        try:
            return f"{int(round(float(val))):,}"
        except Exception:
            return str(val)
    env.filters['intcomma'] = intcomma
    env.globals.update({
        'sxi_data': {},
        'has_master': False,
        'has_eda': False,
    })
    return env
