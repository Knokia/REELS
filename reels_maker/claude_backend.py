"""Адаптер Claude API под интерфейс локальной llama-cpp модели.

pipeline.py везде вызывает LLM как self._llm(prompt, max_tokens=..., temperature=...,
stop=[...], echo=False) и читает ответ как out['choices'][0]['text'] /
out['choices'][0].get('finish_reason') — это формат ответа llama-cpp-python
(text-completion). ClaudeBackend.__call__ воспроизводит тот же контракт, чтобы
find_highlights/generate_clip_title/generate_hook/analyze_zoom_points в
pipeline.py вообще не знали, с каким бэкендом говорят — только load_llm()
решает, что создавать.

ВАЖНО (статус на 2026-07-23): это заготовка, ни разу не проверенная реальным
вызовом API — у пользователя нет рабочего доступа к Claude API с российского
IP (см. обсуждение). Логика прогнана только на уровне unit-тестов с
замоканным клиентом (regex ChatML->plain text, обрезка по stop, подсчёт
стоимости). Перед первым реальным использованием стоит явно проверить один
живой вызов и свериться с полем usage в ответе.
"""
import re
import time

# Локальная модель получает уже готовый ChatML-текст (см. ProcessingThread.
# _chat_prompt: "<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n"),
# т.к. llama-cpp-python работает в режиме text-completion. Claude — обычный
# chat-API (messages=[...]), поэтому конверт снимается обратно в текст
# сообщения пользователя перед отправкой.
_CHATML_RE = re.compile(
    r'^<\|im_start\|>user\n(.*)<\|im_end\|>\n<\|im_start\|>assistant\n$',
    re.DOTALL,
)

# Эти два стоп-токена существуют только в ChatML-формате локальной модели —
# Claude их никогда не сгенерирует, передавать их в stop_sequences API
# бессмысленно (но и не вредно; фильтруем просто для чистоты).
_LOCAL_ONLY_STOPS = {"<|im_end|>", "<|endoftext|>"}


class ClaudeBackend:
    """Копит суммарную стоимость прогона в self.total_cost_usd — pipeline.py
    логирует это перед освобождением LLM в конце run()."""

    # $ за миллион токенов (вход, выход). Источник: platform.claude.com/docs/
    # en/about-claude/pricing, сверено 2026-07-23. Sonnet 5 — цена интро-периода
    # до 2026-09-01, дальше $3/$15. Обновить при смене модели/цен.
    PRICING = {
        "claude-haiku-4-5-20251001": (1.00, 5.00),
        "claude-sonnet-5":           (2.00, 10.00),
    }

    def __init__(self, api_key: str, model: str, log_cb=None):
        import anthropic  # ленивый импорт — пакет нужен, только если реально выбран этот бэкенд
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.log_cb = log_cb or (lambda *_: None)
        self.total_cost_usd = 0.0
        # Официальный контекст Claude 4.x/5 — 200K токенов (у части моделей
        # доступен расширенный 1M, но не полагаемся на это по умолчанию).
        # Используется только для расчёта бюджета обрезки в _fit_text_to_context —
        # реального смысла "закончится контекст" в этом проекте почти не бывает
        # (даже часовое видео — это ~20-25К токенов транскрипта).
        self._n_ctx = 200_000

    def n_ctx(self) -> int:
        return self._n_ctx

    # ── "Токенизация" только для _fit_text_to_context в pipeline.py ────
    # Anthropic SDK не даёт офлайн-токенайзер, совместимый по словарю с
    # реальным биллингом — здесь это ЕДИНИЦА ОБРЕЗКИ текста (нужно только
    # tokenize()+detokenize() быть взаимно обратимыми на префиксе), а НЕ
    # точная оценка числа токенов для расчёта стоимости. Реальные input_tokens/
    # output_tokens берутся из usage в ответе API — см. __call__.
    _CHUNK_BYTES = 3

    def tokenize(self, data: bytes, add_bos: bool = False) -> list:
        n = self._CHUNK_BYTES
        return [data[i:i + n] for i in range(0, len(data), n)]

    def detokenize(self, tokens: list) -> bytes:
        return b"".join(tokens)

    def __call__(self, prompt: str, max_tokens: int = 200, temperature: float = 0.7,
                 top_p: float = 1.0, stop=None, echo: bool = False) -> dict:
        m = _CHATML_RE.match(prompt)
        user_content = m.group(1) if m else prompt

        stop_sequences = [s for s in (stop or []) if s not in _LOCAL_ONLY_STOPS]

        kwargs = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_content}],
        )
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        if top_p and top_p < 1.0:
            kwargs["top_p"] = top_p

        last_err = None
        resp = None
        for attempt in range(3):
            try:
                resp = self._client.messages.create(**kwargs)
                break
            except Exception as e:
                last_err = e
                self.log_cb(f"⚠️ Claude API: попытка {attempt + 1}/3 не удалась ({e})")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        if resp is None:
            raise last_err

        text = "".join(
            block.text for block in resp.content if getattr(block, "type", None) == "text"
        )

        in_tok = resp.usage.input_tokens
        out_tok = resp.usage.output_tokens
        price_in, price_out = self.PRICING.get(self.model, (0.0, 0.0))
        self.total_cost_usd += (in_tok / 1_000_000) * price_in + (out_tok / 1_000_000) * price_out

        finish_reason = "length" if resp.stop_reason == "max_tokens" else "stop"
        return {
            "choices": [{"text": text, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "total_tokens": in_tok + out_tok,
            },
        }
