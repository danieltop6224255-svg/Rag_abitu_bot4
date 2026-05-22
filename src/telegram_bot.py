import os
import time
import requests
from dotenv import load_dotenv
from pyprojroot import here

from src.pipeline import Pipeline, RunConfig


WELCOME_TEXT = "Привет! Я умный помощник, который может ответить на твои вопросы по поступлению во ВШЭ."


class HSEAdmissionTelegramBot:
    def __init__(self):
        load_dotenv()
        self.token = os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN missing")

        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.offset = None
        self.greeted_users = set()

        run_config = RunConfig(is_chat_bot=True)
        self.pipeline = Pipeline(
            root_path=here() / "data" / "test_set",
            subset_name="subset.csv",
            questions_file_name="qa.yaml",
            pdf_reports_dir_name="pdf_docs",
            run_config=run_config,
        )

    def send_message(self, chat_id: int, text: str):
        requests.post(
            f"{self.base_url}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=30,
        )

    def get_updates(self):
        params = {"timeout": 30}
        if self.offset is not None:
            params["offset"] = self.offset

        response = requests.get(
            f"{self.base_url}/getUpdates",
            params=params,
            timeout=35,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("result", [])

    def process_message(self, update: dict):
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = chat.get("id")
        text = message.get("text", "").strip()

        if not chat_id or not text:
            return

        if chat_id not in self.greeted_users:
            self.send_message(chat_id, WELCOME_TEXT)
            self.greeted_users.add(chat_id)

        answer, _references = self.pipeline.answer_question(
            question=text,
            messages_context=[text],
        )
        self.send_message(chat_id, answer)

    def run(self):
        while True:
            try:
                updates = self.get_updates()
                for update in updates:
                    self.offset = update["update_id"] + 1
                    self.process_message(update)
            except Exception as exc:
                print(f"Telegram bot loop error: {exc}")
                time.sleep(2)


if __name__ == "__main__":
    bot = HSEAdmissionTelegramBot()
    bot.run()
