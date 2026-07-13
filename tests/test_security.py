import unittest
from datetime import datetime, timedelta, timezone

from jose import jwt

from app.core.config import settings
from app.core.security import create_access_token, decode_access_token


class TokenSecurityTests(unittest.TestCase):
    def test_access_token_round_trip(self) -> None:
        token = create_access_token("user-id")

        self.assertEqual(decode_access_token(token), "user-id")

    def test_token_without_access_type_is_rejected(self) -> None:
        token = jwt.encode(
            {
                "sub": "user-id",
                "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
                "iat": datetime.now(timezone.utc),
            },
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        with self.assertRaisesRegex(ValueError, "Invalid token payload"):
            decode_access_token(token)

    def test_token_without_expiry_is_rejected(self) -> None:
        token = jwt.encode(
            {
                "sub": "user-id",
                "iat": datetime.now(timezone.utc),
                "type": "access",
            },
            settings.JWT_SECRET_KEY,
            algorithm=settings.JWT_ALGORITHM,
        )

        with self.assertRaisesRegex(ValueError, "Invalid or expired token"):
            decode_access_token(token)


if __name__ == "__main__":
    unittest.main()
