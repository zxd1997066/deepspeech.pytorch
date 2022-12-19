import hydra
import torch
import os

from deepspeech_pytorch.configs.inference_config import EvalConfig
from deepspeech_pytorch.decoder import GreedyDecoder
from deepspeech_pytorch.loader.data_loader import SpectrogramDataset, AudioDataLoader
from deepspeech_pytorch.utils import load_model, load_decoder
from deepspeech_pytorch.validation import run_evaluation


@torch.no_grad()
def evaluate(cfg: EvalConfig):
    device = torch.device("cuda" if cfg.model.cuda else "cpu")

    model = load_model(
        device=device,
        model_path=cfg.model.model_path
    )

    decoder = load_decoder(
        labels=model.labels,
        cfg=cfg.lm
    )
    target_decoder = GreedyDecoder(
        labels=model.labels,
        blank_index=model.labels.index('_')
    )
    test_dataset = SpectrogramDataset(
        audio_conf=model.spect_cfg,
        input_path=hydra.utils.to_absolute_path(cfg.test_path),
        labels=model.labels,
        normalize=True
    )
    test_loader = AudioDataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers
    )
    if cfg.profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            schedule=torch.profiler.schedule(
                wait=int(cfg.num_iter/2),
                warmup=2,
                active=1,
            ),
            on_trace_ready=trace_handler,
        ) as p:
            wer, cer = run_evaluation(
                test_loader=test_loader,
                device=device,
                model=model,
                decoder=decoder,
                target_decoder=target_decoder,
                precision=cfg.model.precision,
                cfg=cfg,
                p=p
            )
    else:
        wer, cer = run_evaluation(
            test_loader=test_loader,
            device=device,
            model=model,
            decoder=decoder,
            target_decoder=target_decoder,
            precision=cfg.model.precision,
            cfg=cfg
        )

    print('Test Summary \t'
          'Average WER {wer:.3f}\t'
          'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))

def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cpu_time_total")
    print(output)
    import pathlib
    timeline_dir = str(pathlib.Path.cwd()) + '/timeline/'
    if not os.path.exists(timeline_dir):
        try:
            os.makedirs(timeline_dir)
        except:
            pass
    timeline_file = timeline_dir + 'timeline-' + str(torch.backends.quantized.engine) + '-' + \
                'deepspeech-' + str(p.step_num) + '-' + str(os.getpid()) + '.json'
    p.export_chrome_trace(timeline_file)
