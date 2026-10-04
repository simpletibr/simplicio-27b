with open("train_simplicio_27b.py", "r", encoding="utf-8") as f:
    code = f.read()

# Add argument for freeze_bottom_layers
old_args = '    parser.add_argument("--max_steps", type=int, default=150, help="Numero maximo de passos de treino")'
new_args = """    parser.add_argument("--max_steps", type=int, default=150, help="Numero maximo de passos de treino")
    parser.add_argument("--freeze_bottom_layers", type=int, default=48, help="Congelar N primeiras camadas (ex: 48 de 64 para focar treino no topo)")
    parser.add_argument("--attention_only_lora", action="store_true", default=False, help="Restringir LoRA apenas a atencao (q_proj, v_proj), preservando MLPs intactos")"""

code = code.replace(old_args, new_args)

# Add special tokens and selective freezing in main()
old_peft = """    # 5. Configurar Adaptadores LoRA
    print(f"Configurando LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = args.lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )"""

new_peft = """    # 5. Registrar Special Tokens de Protocolo e Configurar Adaptadores LoRA
    special_tokens = ["<orient>", "</orient>", "<plan>", "</plan>", "<patch>", "</patch>", "<validate>", "</validate>", "<deliver>", "</deliver>"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    if num_added > 0:
        print(f"Adicionados {num_added} special tokens dedicados para ancoragem de atencao: {special_tokens}")
        model.resize_token_embeddings(len(tokenizer))

    target_mods = ["q_proj", "v_proj", "o_proj"] if args.attention_only_lora else ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    print(f"Configurando LoRA (r={args.lora_r}, alpha={args.lora_alpha}, target_modules={target_mods})...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_r,
        target_modules = target_mods,
        lora_alpha = args.lora_alpha,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
    )

    # Congelamento Seletivo de Camadas (Selective Layer Freezing) para evitar Catastrophic Forgetting
    if args.freeze_bottom_layers > 0:
        print(f"Aplicando Selective Freezing: Congelando as {args.freeze_bottom_layers} primeiras camadas do modelo (0 a {args.freeze_bottom_layers - 1})...")
        frozen_count = 0
        for name, param in model.named_parameters():
            for layer_idx in range(args.freeze_bottom_layers):
                if f"layers.{layer_idx}." in name:
                    param.requires_grad = False
                    frozen_count += 1
                    break
        print(f"Total de tensores congelados nas camadas inferiores: {frozen_count}. Preservando raciocinio base intacto.")"""

code = code.replace(old_peft, new_peft)

with open("train_simplicio_27b.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully enhanced train_simplicio_27b.py with selective layer freezing and special tokens registration!")
