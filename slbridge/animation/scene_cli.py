"""Run fast word composition, optional generation and the local motion library."""
from __future__ import annotations

import argparse
from pathlib import Path
import sqlite3


def main(argv=None):
    parser = argparse.ArgumentParser(description="Free text CSL demonstration workspace")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--presentation", type=Path)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--definitions", type=Path,
                        default=Path(__file__).with_name("basic_scenes_v1.json"))
    parser.add_argument("--port", type=int, default=8879)
    parser.add_argument("--generation-config", type=Path, help="Server-owned JSON paths for the existing SignSparK worker")
    parser.add_argument("--composition-config", type=Path, help="Server-owned word_library and planner JSON configuration")
    parser.add_argument("--recognition-url", help="Loopback v3 recognition API for the communication product")
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        from .registry import MotionRegistry
        from .corpus_store import CorpusStore
        from .presentation import PresentationBundle
        from .scenes import SceneLibrary
        from .scene_routes import mount_scene_routes
        from fastapi import FastAPI
        from .api import install_animation_routes
        import uvicorn

        checksum = args.expected_sha256
        sidecar = Path(str(args.archive) + ".sha256")
        if checksum is None and sidecar.is_file():
            checksum = sidecar.read_text(encoding="utf-8-sig").split()[0]
        registry = MotionRegistry(args.archive, checksum)
        registry.corpus = CorpusStore(args.corpus)
        if args.presentation:
            registry.presentation = PresentationBundle(args.presentation, registry.catalog()["assets"])
        library = SceneLibrary(registry.corpus, args.definitions)
        # Generation-only service: avoid importing and retaining the unrelated
        # recognition runtime while a separate CUDA worker needs commit memory.
        app = FastAPI(title='SignLanguage Bridge text demonstration')
        install_animation_routes(app, registry)
        mount_scene_routes(app, library)
        from .text_generation_jobs import GenerationService
        from .text_generation_routes import mount_text_generation_routes
        root=Path(__file__).resolve().parents[2]
        if args.generation_config:
            import json
            generation_config=json.loads(args.generation_config.read_text(encoding='utf-8'))
        else:
            generation_config={
                'python':'G:/Archive/Zenodo/SignSparK/runtime/Scripts/python.exe',
                'worker_script':str(root/'scripts/animation/generate_custom_signspark.py'),
                'source':'G:/Archive/Zenodo/SignSparK/source',
                'text_assets':'G:/Archive/Zenodo/SignSparK/text-assets',
                'models':'G:/Archive/Zenodo/SignSparK/segmented-checkpoints',
                'base_archive':str(args.archive),
                'source_presentation':'G:/Archive/Zenodo/SOKE/gesture-corpus-100-20261106/presentation',
                'smplx_model':'G:/Archive/Zenodo/SOKE/dependencies/SMPLX_NEUTRAL.npz',
                'output_root':'G:/Archive/Zenodo/SignSparK/web-generation-20261108',
                'readonly_checkpoint_mmap':True,
                'cpu_offload':True,
            }
        generation_service=GenerationService(generation_config)
        mount_text_generation_routes(app,generation_service,home_path='/communicate/')
        from .word_composition_routes import mount_word_composition_routes
        composition_service = None
        if args.composition_config:
            import json
            from .word_composition import WordLibrary, CompositionService
            from .text_gloss_planner import TextGlossPlanner
            composition_config = json.loads(args.composition_config.read_text(encoding='utf-8-sig'))
            word_library = WordLibrary(composition_config['word_library'])
            composition_service = CompositionService(word_library, TextGlossPlanner(composition_config['planner']))
        mount_word_composition_routes(app, composition_service)
        from ..communication import RecognitionGateway
        from ..communication_routes import mount_communication_routes
        recognition_gateway = RecognitionGateway(args.recognition_url) if args.recognition_url else None
        mount_communication_routes(app, recognition_gateway)
    except (OSError, ValueError, KeyError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(f"SignLanguage Bridge: http://127.0.0.1:{args.port}/communicate/ "
          f"(diagnostics: /generate/, /animation/)")
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
