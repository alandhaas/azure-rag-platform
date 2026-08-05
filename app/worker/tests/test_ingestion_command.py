import json

import pytest
from rag_worker.commands import IngestionCommand, IngestionCommandError


def test_ingestion_command_parses_json_payload() -> None:
    command = IngestionCommand.from_json(
        json.dumps(
            {
                "document_id": "doc-123",
                "blob_uri": "http://127.0.0.1:10000/devstoreaccount1/documents/doc-123.pdf",
                "correlation_id": "request-123",
            }
        )
    )

    assert command == IngestionCommand(
        document_id="doc-123",
        blob_uri="http://127.0.0.1:10000/devstoreaccount1/documents/doc-123.pdf",
        correlation_id="request-123",
    )


def test_ingestion_command_parses_bytes_payload() -> None:
    command = IngestionCommand.from_json(
        b'{"document_id":"doc-123","blob_uri":"azurite://documents/doc-123.pdf",'
        b'"correlation_id":"request-123"}'
    )

    assert command.document_id == "doc-123"
    assert command.blob_uri == "azurite://documents/doc-123.pdf"
    assert command.correlation_id == "request-123"


def test_ingestion_command_serializes_to_stable_json() -> None:
    command = IngestionCommand(
        document_id="doc-123",
        blob_uri="azurite://documents/doc-123.pdf",
        correlation_id="request-123",
    )

    assert (
        command.to_json() == '{"blob_uri":"azurite://documents/doc-123.pdf",'
        '"correlation_id":"request-123","document_id":"doc-123"}'
    )


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not-json",
        "[]",
        '{"document_id":"","blob_uri":"azurite://documents/doc.pdf","correlation_id":"req"}',
        '{"document_id":"doc","blob_uri":"","correlation_id":"req"}',
        '{"document_id":"doc","blob_uri":"azurite://documents/doc.pdf","correlation_id":""}',
        '{"document_id":123,"blob_uri":"azurite://documents/doc.pdf","correlation_id":"req"}',
    ],
)
def test_ingestion_command_rejects_invalid_payload(payload: str) -> None:
    with pytest.raises(IngestionCommandError):
        IngestionCommand.from_json(payload)


def test_ingestion_command_rejects_blank_constructor_values() -> None:
    with pytest.raises(IngestionCommandError):
        IngestionCommand(
            document_id=" ",
            blob_uri="azurite://documents/doc.pdf",
            correlation_id="req",
        )
