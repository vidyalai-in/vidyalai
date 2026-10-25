"""Exp 3: three deliberately tricky tool schemas and 48 prompts that tempt violations.

Every schema stays inside the strict-mode JSON Schema subset (additionalProperties: false,
no numeric bounds), so the strict and non-strict arms send byte-identical schemas apart from
the `strict` flag. Numeric ranges live only in descriptions and are checked separately.
"""
import copy

AIRPORT = {"type": "string", "pattern": "^[A-Z]{3}$", "description": "IATA airport code, three capital letters, e.g. SFO"}

BOOK_TRIP = {
    "name": "book_trip",
    "description": ("Book a flight itinerary for one traveler record. Use it for any request to book or reserve flights. "
                    "Dates are ISO 8601 (YYYY-MM-DD). Airports are IATA codes. passengers is the total head count, 1 to 9."),
    "input_schema": {
        "type": "object",
        "properties": {
            "traveler": {
                "type": "object",
                "properties": {
                    "first_name": {"type": "string"},
                    "last_name": {"type": "string"},
                    "loyalty_tier": {"type": "string", "enum": ["none", "silver", "gold", "platinum"]},
                },
                "required": ["first_name", "last_name", "loyalty_tier"],
                "additionalProperties": False,
            },
            "passengers": {"type": "integer", "description": "Total number of travelers, 1 to 9"},
            "cabin": {"type": "string", "enum": ["economy", "premium_economy", "business", "first"]},
            "departure_date": {"type": "string", "format": "date"},
            "legs": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {"from_airport": AIRPORT, "to_airport": AIRPORT, "flexible": {"type": "boolean"}},
                    "required": ["from_airport", "to_airport", "flexible"],
                    "additionalProperties": False,
                },
            },
            "return_date": {"type": "string", "format": "date"},
            "seat_preference": {"type": "string", "enum": ["aisle", "window", "middle"]},
            "meal": {"type": "string", "enum": ["standard", "vegetarian", "vegan", "kosher", "halal", "gluten_free"]},
            "checked_bags": {"type": "integer", "description": "Checked bags per passenger, 0 to 3"},
            "max_price_usd": {"type": "integer", "description": "Upper limit on total fare in whole US dollars"},
            "travel_insurance": {"type": "boolean"},
            "wheelchair_assistance": {"type": "boolean"},
        },
        "required": ["traveler", "passengers", "cabin", "departure_date", "legs"],
        "additionalProperties": False,
    },
}

FILE_BUG = {
    "name": "file_bug",
    "description": ("File a bug in the engineering tracker. priority: P1 is an outage or data loss, P2 a major feature broken, "
                    "P3 a minor issue, P4 cosmetic. severity is an integer from 1 (lowest) to 5 (highest)."),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "priority": {"type": "string", "enum": ["P1", "P2", "P3", "P4"]},
            "severity": {"type": "integer", "description": "1 (lowest) to 5 (highest)"},
            "component": {"type": "string", "enum": ["Billing API", "Web App", "Mobile App", "Data Pipeline", "Auth Service"]},
            "labels": {"type": "array", "items": {"type": "string", "enum": ["regression", "security", "performance", "ux", "billing", "flaky-test"]}},
            "reporter": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "pattern": "^CUS-[0-9]{6}$", "description": "Like CUS-004211"},
                    "plan": {"type": "string", "enum": ["free", "pro", "enterprise"]},
                },
                "required": ["customer_id", "plan"],
                "additionalProperties": False,
            },
            "assignee_email": {"type": ["string", "null"], "format": "email"},
            "due_date": {"type": "string", "format": "date"},
            "affected_users": {"type": "integer", "description": "Estimated number of affected users"},
        },
        "required": ["title", "priority", "severity", "component", "labels", "reporter", "assignee_email"],
        "additionalProperties": False,
    },
}

SCHEDULE_REPORT = {
    "name": "schedule_report",
    "description": ("Schedule a recurring analytics report. hour is 0 to 23 in UTC, minute must be 0, 15, 30 or 45. "
                    "Recipients are email addresses."),
    "input_schema": {
        "type": "object",
        "properties": {
            "report": {"type": "string", "enum": ["weekly_revenue", "daily_signups", "churn_cohorts", "support_volume"]},
            "days": {"type": "array", "minItems": 1, "items": {"type": "string", "enum": ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]}},
            "hour": {"type": "integer", "description": "0 to 23, UTC"},
            "minute": {"type": "integer", "enum": [0, 15, 30, 45]},
            "format": {"type": "string", "enum": ["pdf", "csv", "xlsx"]},
            "recipients": {"type": "array", "minItems": 1, "items": {"type": "string", "format": "email"}},
            "filters": {
                "type": "object",
                "properties": {
                    "region": {"type": "string", "enum": ["NA", "EMEA", "APAC", "LATAM"]},
                    "min_plan": {"type": "string", "enum": ["free", "pro", "enterprise"]},
                },
                "required": ["region"],
                "additionalProperties": False,
            },
            "include_charts": {"type": "boolean"},
        },
        "required": ["report", "days", "hour", "minute", "format", "recipients"],
        "additionalProperties": False,
    },
}

TOOLS = [BOOK_TRIP, FILE_BUG, SCHEDULE_REPORT]


def tools(strict):
    out = copy.deepcopy(TOOLS)
    if strict:
        for t in out:
            t["strict"] = True
    return out


# Ranges stated only in descriptions (strict mode cannot enforce numeric bounds).
RANGES = {
    "book_trip": {"passengers": (1, 9), "checked_bags": (0, 3)},
    "file_bug": {"severity": (1, 5)},
    "schedule_report": {"hour": (0, 23)},
}

SYSTEM = ("You are an operations assistant. Today is Saturday 2026-10-24. "
          "Act on each request by calling exactly one tool. Do not ask follow-up questions: "
          "if something is unclear, make your best reading of the request and call the tool.")

# (expected tool, prompt). Each one leans on a known weak spot: words for numbers, synonyms
# for enum values, relative dates, city names instead of codes, extra details with no field,
# out-of-range values, missing required data.
PROMPTS = [
    ("book_trip", "Book me (Dana Whitfield, gold member) on a flight from San Francisco to New York next Friday, business class."),
    ("book_trip", "Two adults and a toddler, SFO to Lisbon on the 3rd of December, cheapest cabin. Name's Arjun Mehta, no loyalty status."),
    ("book_trip", "Fly Priya Nair from Boston Logan to Heathrow then on to Delhi, Dec 12, premium economy, she's platinum. Veggie meal, window seat."),
    ("book_trip", "I need a dozen seats from ORD to Las Vegas on 2026-11-14 for our offsite, economy, booked under Sam Ortiz (silver)."),
    ("book_trip", "Get Lena Kraus a first class ticket Frankfurt to Tokyo Haneda, leaving Nov 2 and returning two weeks later. She'll bring 5 checked bags."),
    ("book_trip", "Book a flight to Paris for tomorrow, economy, one person, Tom Becker."),
    ("book_trip", "Reserve biz class SEA->SFO on Halloween for Kim Lee (gold). Budget is about $1.2k, and add insurance."),
    ("book_trip", "Please book Omar Haddad, 1 pax, economy plus, from Dubai to Cairo on 11/20/2026, halal meal, needs a wheelchair at the gate."),
    ("book_trip", "Book Mei Chen from LAX to Honolulu on 2026-12-20, economy, 2.5 passengers (two adults and a lap infant), aisle seat, no status."),
    ("book_trip", "Flight for Jonas Berg (Platinum Elite) from Stockholm Arlanda to New York JFK, first week of December, cabin: Business. Note: he is allergic to peanuts."),
    ("book_trip", "Book a round trip, Atlanta to Mexico City, leaving 2026-11-05 and back 2026-11-09, for Carla Ruiz, economy, flexible dates on the way out only."),
    ("book_trip", "Need ten people on a flight Denver to Austin next Wednesday, economy, under Greg Hall, no status, max price two thousand dollars."),
    ("book_trip", "Book Ana Souza from São Paulo GRU to Lisbon, 2026-12-01, cabin: coach, gluten-free meal, and three bags each."),
    ("book_trip", "Book Felix Wong, gold tier, from HKG to SIN to SYD on Nov 30, first class, travel insurance yes, wheelchair no."),
    ("book_trip", "Put Nadia Petrova on the earliest flight from Moscow to Istanbul on 12/24, economy, she's a 'Diamond' member."),
    ("book_trip", "Hey, book Rob Stone a trip: Chicago to Miami, Thanksgiving Day 2026, economy, 0 bags, aisle, 1 traveler, no loyalty."),
    ("file_bug", "Checkout is completely down for all EU users since 9am, payments failing with 500s from the billing API. Reported by CUS-004211 (enterprise). Assign to oncall@acme.io."),
    ("file_bug", "Minor: the logout button on the web app is misaligned by 2px. Customer CUS-118230 on the free plan noticed. Nobody assigned yet."),
    ("file_bug", "Mobile app crashes on launch for iOS 19 users, looks like a regression from Tuesday's release. Severity: critical. Reporter customer 77812, pro plan. Assign to maria@acme.io, due next Friday."),
    ("file_bug", "The nightly data pipeline job is flaky, fails about one run in five. Low priority, severity two-ish. Reported by CUS-000932 (pro). Labels: flaky test, performance."),
    ("file_bug", "Security issue: password reset tokens in the auth service never expire. Urgent! Severity 10/10. From CUS-550021 enterprise. Assign security team (sec@acme.io). About 40k users affected."),
    ("file_bug", "Invoices show the wrong currency symbol for JPY in the Billing API. P3, sev 3, labels billing and UX. Reporter: CUS-202020, Enterprise plan. Unassigned. Due 2026-11-15."),
    ("file_bug", "Web app dashboard takes 30 seconds to load for big accounts. High priority, severity 4, performance label, reported by cus-778899 on pro, assign to perf@acme.io, roughly a few hundred users hit it."),
    ("file_bug", "Typo on the pricing page ('Entreprise'). Cosmetic. Severity 1. Reporter CUS-310001, free plan, no assignee, labels: ux, typo."),
    ("file_bug", "After the last deploy, the mobile app logs users out every 5 minutes. P2, severity four. Reporter CUS-640022 (Pro). Assign to ken@acme, labels regression and auth."),
    ("file_bug", "Data pipeline dropped 3 days of events for EMEA customers: data loss. Highest priority, severity 5, reporter CUS-090909 on enterprise, assignee data-oncall@acme.io, due tomorrow, about 12,000 affected users."),
    ("file_bug", "Billing API double-charges when a card is retried. Customer CUS-123456 (enterprise) reported it. Priority: blocker. Severity: high. Assign to billing-team@acme.io."),
    ("file_bug", "Login with Google fails intermittently in the Auth Service, maybe 1% of attempts. Medium priority. Severity 3. Reporter CUS-333444, plan: business. No assignee."),
    ("file_bug", "Dark mode colors are off on the web app settings page. P4. Severity: zero, it's purely cosmetic. Reporter CUS-454545 free. Unassigned."),
    ("file_bug", "Mobile push notifications arrive twice. P3, severity 2, labels ux and regression. Reporter is customer CUS-12345 on pro. Assign to push@acme.io. Affects 'a lot' of users."),
    ("file_bug", "The CSV export in the web app truncates rows after 65,536. P2, sev 3. Reporter CUS-808080, enterprise. Assign to exports@acme.io and also cc lead@acme.io."),
    ("file_bug", "Flaky test in the billing API suite: test_refund_rounding fails randomly on CI. P4, severity 1, label flaky-test, reporter is internal (use CUS-000000, free). Unassigned."),
    ("schedule_report", "Send the weekly revenue report every Monday at 9am UTC as a PDF to cfo@acme.io and finance@acme.io."),
    ("schedule_report", "Daily signups every weekday at 8:10 in the morning UTC, csv, to growth@acme.io, EMEA only."),
    ("schedule_report", "Churn cohorts report on the first and fifteenth of each month at noon, Excel format, to analytics@acme.io."),
    ("schedule_report", "Support volume, weekends only, at 6pm UTC, PDF with charts, to support-leads@acme.io, APAC region, pro plans and up."),
    ("schedule_report", "Weekly revenue every Friday at 25:00 UTC as xlsx for ceo@acme.io, North America only."),
    ("schedule_report", "Daily signups every day at half past seven in the evening UTC, csv, to marketing@acme.io and the growth team."),
    ("schedule_report", "Schedule churn cohorts on Tuesdays and Thursdays, 14:45 UTC, PDF, to retention@acme.io, region: Europe, no charts."),
    ("schedule_report", "Support volume report Mon/Wed/Fri at 7 AM Pacific time, as a Word document, to ops@acme.io."),
    ("schedule_report", "Weekly revenue, sundays 23:59 UTC, csv, to board@acme.io, LATAM, enterprise only, include charts."),
    ("schedule_report", "Daily signups at 0600 UTC every day of the week in PDF to Jane at jane@acme.io and to bob at acme dot io."),
    ("schedule_report", "Send churn cohorts every weekday morning at nine, xlsx, to cs@acme.io, all regions."),
    ("schedule_report", "Support volume every Saturday at 10:20 UTC, CSV, recipients: help@acme.io; region apac; min plan Free."),
    ("schedule_report", "Weekly revenue Mondays at 9, PDF, to finance@acme.io, region: Global."),
    ("schedule_report", "Run the daily signups report hourly on weekdays as csv to data@acme.io."),
    ("schedule_report", "Churn cohorts, every Wednesday at 4:30 PM UTC, pdf, to vp-cs@acme.io, EMEA, minimum plan pro, charts on."),
    ("schedule_report", "Support volume, Thursdays, 11:15 UTC, xlsx, to support@acme.io, Latin America, enterprise customers only."),
]
assert len(PROMPTS) == 48
