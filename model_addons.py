import adapters
from transformers.modeling_outputs import BaseModelOutputWithPooling
from adapters.heads import PredictionHead
import torch
from torch import nn

# create a custom head class by inheriting from the original PredictionHead class
# it is necessary to run paligemma as is, since the model isn't properly supported by adapters library

class IdentityHead(PredictionHead):
    def __init__(self):
        super().__init__(name="identity_head")
        self.config = {
            "layers": 1,
            "activation_function": None,
            "use_pooler": False,
            "dropout_prob": 0.0
        }
        self.identity = nn.Identity()
        # add the identity module
        self.add_module("0", self.identity)

    def build(self, model):
        # override build to do nothing since we just want identity functionality
        self.train(model.training)  # make sure training mode is consistent

    def forward(self, hidden_states, **kwargs):
        # ensure we maintain the correct dimensions
        # hidden_states shape: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]

        # check if we need to preserve dimensions
        original_shape = hidden_states.shape

        # Apply identity transformation (maintaining the original shape)
        output = super().forward(hidden_states)

        # Ensure output has the same shape as input
        if output.shape != original_shape:
            output = output.view(original_shape)

        return output

    def get_label_names(self):
        # Override to return the expected label names
        return ["labels"]

def dummy_save_all_heads(self, *args, **kwargs):
    # this dummy method does nothing, but prevents AttributeError during training.
    pass
    
def generate_caption(model, processor, input_text, input_image):
    inputs = processor(text=input_text, 
                       images=input_image,                      
                       padding="longest", 
                       do_convert_rgb=True, 
                       return_tensors="pt").to(device)
    
    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs["pixel_values"].to(device)
    input_len = input_ids.shape[1]
    generated = input_ids.clone()
    max_new_tokens = 32
 
    with torch.no_grad():
        past_key_values = None
        for _ in range(max_new_tokens):
            model_inputs = {"input_ids": generated}
            if past_key_values is not None:
                model_inputs["past_key_values"] = past_key_values
            if generated.shape[1] == input_len:  # Only pass pixel_values on first step
                model_inputs["pixel_values"] = pixel_values
            out = model(**model_inputs)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            past_key_values = out.past_key_values
            if next_token.item() == processor.tokenizer.eos_token_id:
                break
        # Decode generated tokens (skip input prompt)
        caption_ids = generated[0, input_len:]
        caption = processor.decode(caption_ids, skip_special_tokens=True)
        return caption.strip()

