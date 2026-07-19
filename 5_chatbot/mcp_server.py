import os
import requests
from mcp.server.fastmcp import FastMCP

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY")

mcp = FastMCP("calc-stock-server")


@mcp.tool()
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "zero division error"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation {operation}"}

        return {"first_num": first_num, "second_num": second_num, "result": result}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {"error": "ALPHA_VANTAGE_API_KEY not set"}

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY&symbol={symbol}"
        f"&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(transport="stdio")