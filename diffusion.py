class VDiffusionInpainting(VDiffusion):
   

    def forward(self, x: Tensor, xpre : Tensor, **kwargs) -> Tensor:  # type: ignore
        batch_size, device = x.shape[0], x.device
        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        # Get noise
        noise = torch.randn_like(x)
        # Combine input and noise weighted by half-circle
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        x_noisy = torch.cat([xpre, x_noisy], dim=-1)
        v_target = alphas * noise - betas * x
        # Predict velocity and return loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        v_pred = v_pred[:,:,xpre.shape[-1]:]
        return self.loss_fn(v_pred, v_target)