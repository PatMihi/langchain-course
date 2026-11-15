from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable
from pydantic import ValidationError

MAX_ITERATIONS = 10
MODEL = "openai:gpt-4.1-mini"  # "llama3.2:latest"


# --- Tools (LangChain @tool decorator) ---


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# --- Agent Loop ---


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}

    llm = init_chat_model(MODEL, temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. Only ask the user a question if the discount tier is "
                "completely missing from their message. If they mention a tier "
                "(e.g. 'gold discount'), that counts as specified — do NOT ask "
                "for confirmation.\n"
                "5. Use the product name exactly as the user wrote it as the "
                "'product' argument (e.g. 'laptop'). Do NOT ask for a specific "
                "model or extra details — just call get_product_price.\n"
                "6. Call only ONE tool per turn. Never put instructions or "
                "text in an argument; arguments must be real values."
            )
        ),
        HumanMessage(content=question),
    ]

    had_error = False
    nudged = False

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # Small models sometimes answer in text right after a tool error
        # (e.g. "please call X") instead of making the tool call. Nudge once.
        if not tool_calls and had_error and not nudged:
            nudged = True
            print("  [Nudge] No tool call after an error; asking the model to call the tool")
            messages.append(ai_message)
            messages.append(
                HumanMessage(
                    content=(
                        "Do not reply with text. You must call the get_product_price "
                        "tool yourself right now using a tool call, then continue."
                    )
                )
            )
            continue

        # If no tool calls, this is the final answer
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        try:
            observation = tool_to_use.invoke(tool_args)
        except ValidationError as e:
            # Feed the error back to the model so it can correct itself
            observation = (
                f"ERROR: invalid arguments for {tool_name}: {e}\n"
                "Arguments must be real values, not instructions. "
                "If you need a value from another tool (e.g. a price), "
                "call that tool first, wait for its result, then use the "
                "exact value it returned."
            )
            had_error = True
            print(f"  [Tool Error] {e}")

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")
