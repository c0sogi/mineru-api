import io
import os
import zipfile
from base64 import b64encode
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple, Optional, cast
from uuid import uuid4

import magic_pdf.model as model_config
from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile
from gotenberg_client import GotenbergClient
from magic_pdf.config.enums import SupportedPdfParseMethod
from magic_pdf.data.data_reader_writer import FileBasedDataWriter
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.operators.models import InferenceResult
from magic_pdf.operators.pipes import PipeResult
from pydantic import BaseModel

OfficeExts = Literal[".docx", ".doc", ".odt", ".pptx", ".ppt", ".odp", ".xls", ".xlsx", ".ods"]

TMP_DIR = "/tmp/{uuid}"
model_config.__use_inside_model__ = True
model_config.__model_mode__ = "full"

app = FastAPI()
parse_router = APIRouter()


@app.get("/health")
def health():
    return {"status": "ok"}


@dataclass
class OfficeConverter:
    gotenberg_url: str = field(default=os.getenv("GOTENBERG_URL", "http://localhost:3500"))
    headers: dict[str, str] = field(default_factory=dict)

    def convert(self, office_file_path: Path | str, output_file_path: Path | str):
        office_file_path = Path(office_file_path)
        output_file_path = Path(output_file_path)
        with GotenbergClient(self.gotenberg_url) as client:
            if self.headers:
                client.add_headers(self.headers)
            with client.libre_office.to_pdf() as route:
                response = route.convert(office_file_path).run()
                output_file_path.write_bytes(response.content)


class ParseResponse(BaseModel):
    md: str
    zip: Optional[bytes] = None


class FileContext(NamedTuple):
    file: UploadFile
    content_dir: Path

    @property
    def filename(self) -> str:
        if filename := self.file.filename:
            return filename
        raise HTTPException(status_code=400, detail="Invalid file name")

    @property
    def stem(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def ext(self) -> str:
        return os.path.splitext(self.filename)[1].removeprefix(".").lower()

    @property
    def files(self) -> dict[Path, bytes]:
        return {
            filepath.relative_to(self.content_dir): filepath.read_bytes() for filepath in self.content_dir.rglob("*")
        }

    async def read_pdf(self) -> bytes:
        file = self.file
        ext = self.ext
        stem = self.stem
        filename = self.filename
        if ext in OfficeExts.__args__:
            contents: bytes = await file.read()
            input_file: Path = self.content_dir / filename
            input_file.write_bytes(contents)
            output_file: Path = self.content_dir / f"{stem}.pdf"
            OfficeConverter().convert(office_file_path=input_file, output_file_path=output_file)
            pdf_bytes = output_file.read_bytes()
        elif ext == "pdf":
            pdf_bytes = await file.read()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        return pdf_bytes

    @classmethod
    def from_file(cls, file: UploadFile) -> "FileContext":
        content_dir = Path(TMP_DIR.format(uuid=uuid4().hex))
        content_dir.mkdir(parents=True, exist_ok=True)
        return cls(file=file, content_dir=content_dir)


def create_b64_zipbytes(contents: dict[Path, bytes]) -> Optional[bytes]:
    if not contents:
        return None
    with io.BytesIO() as mem_zip:
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path, content in contents.items():
                zf.writestr(str(path), content)
        mem_zip.seek(0)
        value = mem_zip.getvalue()
        return b64encode(value)


async def inference_as_markdown(context: FileContext) -> str:
    content_dir: str = context.content_dir.as_posix()
    image_writer: FileBasedDataWriter = FileBasedDataWriter(content_dir)

    ds = PymuDocDataset(await context.read_pdf())
    if ds.classify() == SupportedPdfParseMethod.OCR:
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    infer_result = cast(InferenceResult, infer_result)
    pipe_result = cast(PipeResult, pipe_result)
    return pipe_result.get_markdown(content_dir)

    ### draw model result on each page
    # infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))

    ### get model inference result
    # model_inference_result = infer_result.get_infer_res()

    # ### draw layout result on each page
    # pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))

    # ### draw spans result on each page
    # pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    # ### get markdown content
    # md_content = pipe_result.get_markdown(image_dir)

    # ### dump markdown
    # pipe_result.dump_md(md_writer, f"{name_without_suff}.md", image_dir)

    # ### get content list content
    # content_list_content = pipe_result.get_content_list(image_dir)

    # ### dump content list
    # pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json", image_dir)

    # ### get middle json
    # middle_json_content = pipe_result.get_middle_json()

    # ### dump middle json
    # pipe_result.dump_middle_json(md_writer, f"{name_without_suff}_middle.json")

    # return (md_content, content_list_content)


@parse_router.post("/parse")
async def parse(file: UploadFile = File(...)) -> ParseResponse:
    context: FileContext = FileContext.from_file(file)
    markdown_str: str = await inference_as_markdown(context)
    maybe_zip_bytes: Optional[bytes] = create_b64_zipbytes(context.files)
    return ParseResponse(md=markdown_str, zip=maybe_zip_bytes)


app.include_router(parse_router, prefix="/api")
