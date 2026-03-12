How does Langfuse work? 

Langfuse automatically tracks your agent's token usage, costs, and latency. 
You just need to set up the environment variables (LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST), 
use the @observe() decorator and CallbackHandler() in your code, and generate a unique session ID for each run. 

Langfuse handles the rest. For the full setup and code examples, 
refer to "Resource Management & Toolkit for the Challenge" in the Learn & Train section.