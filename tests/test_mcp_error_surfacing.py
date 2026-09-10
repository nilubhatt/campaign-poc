"""
Adversarial review finding, verified by reproducing it over a real MCP streamable-HTTP
client: a bare ValueError raised from inside a tool function's body (as opposed to a
pydantic schema-validation failure, which the MCP framework already handles fine) is
swallowed by the framework and replaced with a generic "Error executing tool X" -
UnexpectedToolError deliberately withholds the original text. The actual message (e.g.
"tag 'performed_well' cannot be marked source='verified' - add real metrics first...") never
reaches the caller. This defeats the point of every carefully-written validation message in
this codebase - store.py/core.py raise ValueError for exactly these cases. Tools must catch
it themselves and return {"error": str(exc)} - the same convention already used for the
not-found cases (e.g. "campaign X not found").
"""
import mcp_server as m


def test_verified_tag_without_metrics_returns_an_error_dict_not_a_crash(conn):
    result = m.upload_campaign(
        title="X", tags=[{"value": "performed_well", "source": "verified"}], confirm=True)
    assert "error" in result
    assert "verified" in result["error"]
    assert "metric" in result["error"].lower()


def test_update_campaign_verified_tag_without_metrics_returns_an_error_dict(conn):
    created = m.upload_campaign(title="X", confirm=True)
    result = m.update_campaign(
        campaign_id=created["campaign_id"],
        tags=[{"value": "performed_well", "source": "verified"}])
    assert "error" in result


def test_invalid_metric_type_returns_an_error_dict(conn):
    created = m.upload_campaign(title="X", confirm=True)
    result = m.add_metrics(campaign_id=created["campaign_id"], detail="x",
                           metric_type="definitely_not_valid", confirm=True)
    assert "error" in result


def test_nonexistent_campaign_id_find_similar_returns_an_error_dict_not_a_crash(conn):
    result = m.find_similar_campaigns(campaign_id="does-not-exist")
    assert "error" in result


def test_successful_call_is_unaffected(conn):
    result = m.upload_campaign(title="X", detail="brief", confirm=True)
    assert "campaign_id" in result
    assert "error" not in result
