import numpy as np
import unittest
from PIL import Image

from feature_backends import (
    EmbeddingBackend,
    EmbeddingRelevanceScorer,
    FeatureBackendError,
    MepEmbeddingBackend,
    MultipartEmbeddingBackend,
    PanguEmbeddingBackend,
    create_relevance_scorer,
)


class StaticBackend(EmbeddingBackend):
    name = "static"

    def __init__(self, text, images):
        self.text = np.asarray(text, dtype=np.float32)
        self.images = np.asarray(images, dtype=np.float32)

    def embed_texts(self, texts):
        return self.text

    def embed_images(self, images):
        return self.images[: len(images)]


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def raise_for_status(self):
        return None

    def json(self):
        return self.value


class MultipartSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        data = kwargs["data"]
        if "text" in data:
            return FakeResponse({"data": {"embedding": [1.0, 0.0]}})
        return FakeResponse({"data": {"embedding": [0.0, 1.0]}})


class MepSession:
    def __init__(self):
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        task = kwargs["json"]["data"]["task"]
        content = (
            {"text_embedding": [1.0, 0.0]}
            if task == "text_embedding"
            else {"image_embedding": [0.0, 1.0]}
        )
        return FakeResponse(
            {"result": {"code": "0", "content": [content]}}
        )


class JsonStringMepSession(MepSession):
    def post(self, url, **kwargs):
        response = super().post(url, **kwargs)
        response.value["result"]["content"][0] = __import__("json").dumps(
            response.value["result"]["content"][0]
        )
        return response


class CustomScorer:
    def __init__(self, options):
        self.value = float(options["value"])
        self.metadata = {"backend": "custom-test"}

    def prepare_query(self, query):
        self.query = query

    def score_images(self, images):
        return [self.value] * len(images)


class FeatureBackendTests(unittest.TestCase):
    def test_cosine_scorer_matches_original_clip_semantics(self):
        scorer = EmbeddingRelevanceScorer(
            StaticBackend([[3.0, 0.0]], [[2.0, 0.0], [0.0, 4.0]])
        )
        images = [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]

        np.testing.assert_allclose(scorer.score("query", images), [1.0, 0.0])
        self.assertEqual(scorer.metadata["embedding_dimension"], 2)

    def test_scorer_rejects_text_image_dimension_mismatch(self):
        scorer = EmbeddingRelevanceScorer(
            StaticBackend([[1.0, 0.0]], [[1.0, 0.0, 0.0]])
        )

        with self.assertRaisesRegex(FeatureBackendError, "dimensions differ"):
            scorer.score("query", [Image.new("RGB", (2, 2))])

    def test_generic_multipart_backend_uses_configurable_protocol(self):
        session = MultipartSession()
        backend = MultipartEmbeddingBackend(
            {
                "base_url": "http://embedding.test",
                "response_embedding_paths": ["data.embedding"],
                "api_key": "test-token",
                "max_retries": 0,
            },
            session=session,
        )

        self.assertEqual(backend.embed_texts(["hello"]).tolist(), [[1.0, 0.0]])
        self.assertEqual(
            backend.embed_images([Image.new("RGB", (2, 2))]).tolist(), [[0.0, 1.0]]
        )
        self.assertEqual(
            session.calls[0][1]["headers"]["Authorization"], "Bearer test-token"
        )
        self.assertEqual(session.calls[0][1]["files"], {})
        self.assertIn("instruction", session.calls[0][1]["data"])
        self.assertIn("image", session.calls[1][1]["files"])
        self.assertIn("instruction", session.calls[1][1]["data"])

    def test_pangu_backend_matches_pangu_sim_request_defaults(self):
        session = MultipartSession()
        backend = PanguEmbeddingBackend(
            {
                "base_url": "http://pangu.test",
                "api_key": "test-token",
                "response_embedding_paths": ["data.embedding"],
                "max_retries": 0,
            },
            session=session,
        )

        backend.embed_texts(["hello"])
        backend.embed_images([Image.new("RGB", (2, 2))])

        expected_instruction = "Retrieve relevant documents for the user's query."
        self.assertEqual(session.calls[0][1]["data"]["instruction"], expected_instruction)
        self.assertEqual(session.calls[1][1]["data"]["instruction"], expected_instruction)
        image_part = session.calls[1][1]["files"]["image"]
        self.assertEqual(image_part[2], "image/png")

    def test_mep_backend_wraps_requests_and_supports_image_base64(self):
        session = MepSession()
        backend = MepEmbeddingBackend(
            {
                "elb": "http://mep.test/service",
                "appid": "test-app",
                "secret_key": "test-secret",
                "b_id": "test-business",
                "flow_id": "test-flow",
                "max_retries": 0,
            },
            session=session,
        )

        self.assertEqual(backend.embed_texts(["hello"]).tolist(), [[1.0, 0.0]])
        self.assertEqual(
            backend.embed_images([Image.new("RGB", (2, 2))]).tolist(), [[0.0, 1.0]]
        )
        self.assertEqual(session.calls[0][1]["json"]["meta"]["bId"], "test-business")
        self.assertTrue(session.calls[1][1]["json"]["data"]["image"])
        self.assertNotIn("test-secret", str(session.calls))

    def test_mep_backend_accepts_json_encoded_content(self):
        backend = MepEmbeddingBackend(
            {
                "elb": "http://mep.test/service",
                "appid": "test-app",
                "secret_key": "test-secret",
                "b_id": "test-business",
                "flow_id": "test-flow",
                "max_retries": 0,
            },
            session=JsonStringMepSession(),
        )

        self.assertEqual(backend.embed_texts(["hello"]).tolist(), [[1.0, 0.0]])

    def test_missing_embedding_error_includes_mep_status_message(self):
        from feature_backends import _first_path

        with self.assertRaisesRegex(
            FeatureBackendError, "code='400'.*msg='unsupported image task'"
        ):
            _first_path(
                {"code": 400, "msg": "unsupported image task"},
                ["image_embedding"],
                "MEP image embedding",
            )

    def test_python_plugin_can_supply_a_direct_scorer(self):
        scorer = create_relevance_scorer(
            "python",
            {
                "class_path": "test_feature_backends:CustomScorer",
                "options": {"value": 0.75},
            },
            model_name="unused",
            device="cpu",
            batch_size=2,
        )
        scorer.prepare_query("hello")

        self.assertEqual(
            scorer.score_images([Image.new("RGB", (2, 2))]), [0.75]
        )


if __name__ == "__main__":
    unittest.main()
