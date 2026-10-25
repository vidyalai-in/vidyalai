from jsonschema import Draft202012Validator, FormatChecker
from exp3_schemas import SCHEDULE_REPORT

# A real call from the strict-off run: "Daily signups every weekday at 8:10 in the morning UTC"
call = {"report": "daily_signups", "days": ["mon", "tue", "wed", "thu", "fri"], "hour": 8, "minute": 10,
        "format": "csv", "recipients": ["growth@acme.io"], "filters": {"region": "EMEA"}}
v = Draft202012Validator(SCHEDULE_REPORT["input_schema"], format_checker=FormatChecker())
print([e.message for e in v.iter_errors(call)])
