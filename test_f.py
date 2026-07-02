
class Ev:
    severity = 0.5
    event_type = "Split"
    description = "Desc"
    sentiment = "Pos"
    confidence = 0.9

ev = Ev()
sev_class = "badge-low"
sent_class = "sentiment-positive"
event_date = "2026-07-02"

print(f"""
<div class="card">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.25rem;">
        <div style="font-size: 1.05rem; font-weight: 600; color: #f0f6fc;">{ev.event_type}</div>
        <span class="badge {sev_class}">Severity: {ev.severity:.2f}</span>
    </div>
</div>
""")
