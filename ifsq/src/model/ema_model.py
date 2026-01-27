class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

        for name, buffer in self.model.named_buffers():
            self.shadow[name] = buffer.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

        for name, buffer in self.model.named_buffers():
            if name in self.shadow:
                new_average = (
                    1.0 - self.decay
                ) * buffer.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data
                param.data = self.shadow[name]

        for name, buffer in self.model.named_buffers():
            if name in self.shadow:
                self.backup[name] = buffer.data
                buffer.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data = self.backup[name]

        for name, buffer in self.model.named_buffers():
            if name in self.shadow:
                buffer.data = self.backup[name]

        self.backup = {}
