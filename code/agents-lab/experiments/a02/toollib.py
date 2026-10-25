"""Twenty realistic tool definitions for the token-overhead measurement."""

def t(name, desc, props, required):
    return {"name": name, "description": desc,
            "input_schema": {"type": "object", "properties": props, "required": required}}

S = lambda d: {"type": "string", "description": d}
I = lambda d: {"type": "integer", "description": d}

TOOLS = [
    t("get_weather", "Get the current weather for a city. Returns temperature, conditions and wind. Use it when the user asks about current weather, not forecasts.",
      {"city": S("City name, e.g. Paris"), "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"}}, ["city"]),
    t("get_stock_price", "Get the latest trade price in USD for a ticker on a major US exchange. Use it for current prices only; it returns no company fundamentals.",
      {"ticker": S("Ticker symbol, e.g. AAPL")}, ["ticker"]),
    t("search_docs", "Full-text search over the internal product documentation. Returns up to `limit` page titles with snippets. Use it for how-to questions about the product.",
      {"query": S("Search terms"), "limit": I("Maximum results, 1-20")}, ["query"]),
    t("create_ticket", "Open a support ticket in the help-desk system. Returns the new ticket ID. Use it only when the user explicitly asks to file or escalate an issue.",
      {"title": S("One-line summary"), "body": S("Full description"), "priority": {"type": "string", "enum": ["low", "normal", "high", "urgent"]}}, ["title", "body"]),
    t("get_order", "Look up an order by its ID. Returns status, items, totals and shipping info. Use it when the user quotes an order number.",
      {"order_id": S("Order ID like ORD-12345")}, ["order_id"]),
    t("refund_order", "Issue a full or partial refund for an order. Irreversible. Use only after confirming the amount with the user.",
      {"order_id": S("Order ID"), "amount_cents": I("Refund amount in cents"), "reason": S("Short reason")}, ["order_id", "amount_cents"]),
    t("send_email", "Send an email from the support address. Use for follow-ups the user asked for; never for marketing.",
      {"to": S("Recipient address"), "subject": S("Subject line"), "body": S("Plain-text body")}, ["to", "subject", "body"]),
    t("list_calendar_events", "List calendar events for a user between two dates. Returns title, start, end and attendees for each event.",
      {"user_id": S("User ID"), "start_date": S("ISO date YYYY-MM-DD"), "end_date": S("ISO date YYYY-MM-DD")}, ["user_id", "start_date", "end_date"]),
    t("create_calendar_event", "Create a calendar event and invite attendees. Returns the event ID. Check availability first with list_calendar_events.",
      {"title": S("Event title"), "start": S("ISO datetime"), "duration_minutes": I("Length in minutes"), "attendees": {"type": "array", "items": {"type": "string"}, "description": "Email addresses"}}, ["title", "start", "duration_minutes"]),
    t("convert_currency", "Convert an amount between two currencies at today's mid-market rate. Returns the converted amount and the rate used.",
      {"amount": {"type": "number", "description": "Amount to convert"}, "from_currency": S("ISO 4217 code, e.g. USD"), "to_currency": S("ISO 4217 code, e.g. EUR")}, ["amount", "from_currency", "to_currency"]),
    t("run_sql", "Run a read-only SQL query against the analytics warehouse. Returns at most 100 rows as CSV. Writes are rejected.",
      {"query": S("A single SELECT statement")}, ["query"]),
    t("get_user_profile", "Fetch a customer's profile: name, plan, signup date and account status. Use it before answering account-specific questions.",
      {"user_id": S("Customer ID")}, ["user_id"]),
    t("update_user_plan", "Change a customer's subscription plan. Takes effect at the next billing cycle. Confirm with the user before calling.",
      {"user_id": S("Customer ID"), "plan": {"type": "string", "enum": ["free", "pro", "team", "enterprise"]}}, ["user_id", "plan"]),
    t("translate_text", "Translate text into a target language. Returns the translation only. Use for user-facing text, not code.",
      {"text": S("Text to translate"), "target_language": S("ISO 639-1 code, e.g. de")}, ["text", "target_language"]),
    t("get_shipping_quote", "Quote shipping cost and delivery days for a parcel between two postal codes.",
      {"from_postal": S("Origin postal code"), "to_postal": S("Destination postal code"), "weight_grams": I("Parcel weight in grams")}, ["from_postal", "to_postal", "weight_grams"]),
    t("check_inventory", "Check stock level for a product SKU across warehouses. Returns quantity per warehouse.",
      {"sku": S("Product SKU")}, ["sku"]),
    t("geocode_address", "Turn a street address into latitude and longitude. Returns the best match and a confidence score.",
      {"address": S("Full street address")}, ["address"]),
    t("get_exchange_holidays", "List market holidays for a stock exchange in a given year.",
      {"exchange": {"type": "string", "enum": ["NYSE", "NASDAQ", "LSE", "TSE"]}, "year": I("Four-digit year")}, ["exchange", "year"]),
    t("summarize_url", "Fetch a public web page and return a five-sentence summary of its main content. Fails on pages behind a login.",
      {"url": S("Absolute http(s) URL")}, ["url"]),
    t("log_feedback", "Record a customer's product feedback for the product team. Returns nothing useful to the user.",
      {"user_id": S("Customer ID"), "text": S("Feedback text"), "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}}, ["text"]),
]
assert len(TOOLS) == 20 and len({x["name"] for x in TOOLS}) == 20
