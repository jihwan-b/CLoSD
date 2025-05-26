from pathlib import Path

prompt_path = "custom_t2m/prompt_queue.txt"

print("### RoboGo: Real-time prompt writer (txt mode) started ###")
print("Type a prompt and press Enter. Type 'exit' to quit.\n")

while True:
    user_input = input("[Prompt] > ").strip()
    if user_input.lower() == "exit":
        print("Exiting prompt writer.")
        break
    if user_input:
        with open(prompt_path, "a") as f:
            f.write(user_input + "\n")
        print(f"Appended to prompt_queue.txt: '{user_input}'")
