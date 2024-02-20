import torch
from typing import Any, Dict, Optional

from tqdm import tqdm
from transformers import GenerationMixin, TrainerCallback, TextStreamer
from transformers.utils import ModelOutput

from spihtter.utils import imsave, slugify

from .process_inputs import InputProcessorCache, SpihtInputProcessor


class DummyTokenizerPBar:
    def __init__(self, total):
        self.pb = tqdm(total=total)
        self.curr = 0

    def decode(self, x):
        self.pb.update(len(x) - self.curr)
        self.curr = len(x)
        return ""


class GenerateImageCallback(TrainerCallback):
    def __init__(
        self,
        run_every_steps: int = 100,
        max_length: int = 512,
        spiht_input_processor: SpihtInputProcessor = None,
        output_dir: str = "out/",
        generation_device: Optional[str] = None,
        **generate_kwargs,
    ):
        self.run_every_steps = run_every_steps
        self.max_length = max_length
        self.spiht_input_processor = spiht_input_processor
        self.output_dir = output_dir
        self.generation_device = generation_device
        self.generate_kwargs = generate_kwargs

    def on_step_end(self, args, state, control, model, tokenizer, **kwargs):
        if state.global_step % self.run_every_steps != 0:
            return

        model.eval()

        if self.generation_device is not None:
            old_dev, old_dtype = model.device, model.dtype
            model = model.to(self.generation_device)
            if self.generation_device == "cpu":
                model = model.to(torch.float32)

        past_input_processor_cache = []

        input_ids = self.generate_kwargs["input_ids"]
        b = input_ids.shape[0]

        # progress bar
        if b == 1:
            text_streamer = TextStreamer(DummyTokenizerPBar(self.max_length))
        else:
            text_streamer = None

        generation_kwargs = {
                k:v.to(model.device)
                if isinstance(v, torch.Tensor) else v
                for k,v in self.generate_kwargs.items()
        }

        print("GENERATING IMAGE")
        # to generate a spiht image,
        # all you have to do is call generate, and pass an empty list
        # as the past_input_processor_cache
        outputs = model.generate(
            spiht_input_processor=self.spiht_input_processor,
            past_input_processor_cache=past_input_processor_cache,
            max_length=self.max_length,
            streamer=text_streamer,
            **generation_kwargs,
        )

        prompts = ""
        if tokenizer is not None:
            print("Section of first sequence generated: ", repr(tokenizer.decode(outputs[0][:40])))
            prompts = tokenizer.batch_decode(input_ids)


        for i, cache in enumerate(past_input_processor_cache):
            image = None
            if cache.in_progress_image is not None:
                image = cache.in_progress_image.render()
            elif len(cache.finished_images) > 0:
                image = cache.finished_images[-1].image

            if image is not None:
                prompt = prompts[i]
                imsave(
                    image,
                    f"{self.output_dir}/image-{state.global_step:08}-{slugify(prompt[:20])}.png",
                )
            else:
                print(f"Failed to generation image {i}")

        model.train()

        if self.generation_device is not None:
            model = model.to(old_dtype).to(old_dev)


class SpihtGenerationMixin(GenerationMixin):
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        spiht_input_processor: SpihtInputProcessor = None,
        past_input_processor_cache=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """

        Use the spiht input processor to provide correct spiht metadata ids
        returns {input_ids, spiht_metadata_ids}
        """

        b = input_ids.shape[0]

        if spiht_input_processor is None:
            raise ValueError(
                "The generation kwargs should contain a spiht_input_processor!"
            )

        if past_input_processor_cache is None:
            raise ValueError(
                "you must pass the past_input_processor_cache as an empty list to model.generate"
            )

        # TODO
        # Check position ids on the fly; make sure that the
        # past_input_processor_cache has the same internal position as the input
        # positions

        if len(past_input_processor_cache) == 0:
            for _ in range(b):
                past_input_processor_cache.append(None)

        batch_spiht_metadata_ids = []

        for i, (input_id_sequence, past_cache) in enumerate(
            zip(input_ids, past_input_processor_cache)
        ):
            metadata_ids, cache = spiht_input_processor.process_input_ids_with_cache(
                input_id_sequence,
                past_input_processor_cache=past_cache,
                output_input_processor_cache=True,
            )

            batch_spiht_metadata_ids.append(metadata_ids)
            past_input_processor_cache[i] = cache

        batch_spiht_metadata_ids = torch.stack(batch_spiht_metadata_ids).to(self.device)

        return dict(
            input_ids=input_ids,
            spiht_metadata_ids=batch_spiht_metadata_ids,
        )
