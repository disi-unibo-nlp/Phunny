from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
client = OpenAI(
    api_key=OPENAI_KEY
)

client.batches.cancel("batch_6787d4edaea08190929c7a2f4ef5fc21")