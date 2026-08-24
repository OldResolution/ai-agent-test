import json
import re
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def lookup_order(order_id: str) -> dict:
    """Deterministic order lookup with sanitization and status rules."""
    # 1. Normalize and validate
    normalized = order_id.strip().upper()
    if not re.match(r"^ORD-\d{4}$", normalized):
        return {"found": False, "order_id": normalized, "error": "Invalid order ID format."}
    
    # 2. Load data
    orders_path = _DATA_DIR / "orders.json"
    if not orders_path.exists():
        return {"found": False, "order_id": normalized, "error": "Order database not found."}
        
    with open(orders_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # 3. Find order
    target_order = None
    for o in data.get("orders", []):
        if o.get("order_id") == normalized:
            target_order = o
            break
            
    if not target_order:
        return {"found": False, "order_id": normalized}
        
    # 4. Apply status rules and sanitize
    status = target_order.get("status")
    
    sanitized = {
        "found": True,
        "order_id": target_order.get("order_id"),
        "status": status,
        "status_updated_at": target_order.get("status_updated_at"),
        "carrier": target_order.get("carrier"),
        "tracking_number": target_order.get("tracking_number"),
        "estimated_delivery": target_order.get("estimated_delivery"),
        "shipped_at": target_order.get("shipped_at"),
        "delivered_at": target_order.get("delivered_at"),
        "customer_safe_message": target_order.get("customer_safe_message"),
        "membership_tier": target_order.get("membership_tier"),
    }
    
    # Status rules overrides
    if status in ("cancelled", "returned"):
        sanitized["estimated_delivery"] = None
        sanitized["tracking_number"] = None # Hide tracking predictions for cancelled/returned
        
    elif status == "shipped" and not sanitized.get("estimated_delivery"):
        pass # Allow shipped status and carrier, but no ETA
        
    elif status == "exception":
        sanitized["human_assistance_required"] = True
        
    return {k: v for k, v in sanitized.items() if v is not None}
