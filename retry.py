import time

from logger import log_event


class RetryError(Exception):
    pass


def retry_with_backoff(
    func,
    max_attempts: int = 3,
    delay_seconds: float = 0.1,
    exceptions: tuple = (Exception,),
):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return func()

        except exceptions as error:
            last_error = error

            if attempt == max_attempts:
                break

            log_event("retry", {
                "attempt": attempt,
                "error": str(error),
                "next_delay_seconds": delay_seconds * (2 ** (attempt - 1))
            })

            time.sleep(delay_seconds * (2 ** (attempt - 1)))

    raise RetryError(
        f"Operation failed after {max_attempts} attempts."
    ) from last_error