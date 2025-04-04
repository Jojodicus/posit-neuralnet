#ifndef TRAIN_POSIT_HPP
#define TRAIN_POSIT_HPP

// General headers
#include <cstdint>
#include <iostream>
#include <torch/torch.h>
#include <positnn/positnn>

template <typename Posit, template<typename> class Model, typename DataLoader, typename Optimizer>
void train_posit(	size_t epoch,
					size_t const num_epochs,
					Model<Posit>& model,
					DataLoader& data_loader,
					Optimizer& optimizer,
					size_t const kLogInterval,
					size_t const dataset_size	){

	using Target = unsigned short int;

	model.train();
	size_t batch_idx = 0;
	size_t total_batch_size = 0;

	for(auto const& batch : data_loader) {
		// Update number of trained samples
		size_t const batch_size = batch.target.size(0);
		total_batch_size += batch_size;

		// Get data and target
		auto data_float = batch.data;
		auto target_float = batch.target;

		// Convert data and target to float32 and uint8
		data_float = data_float.to(torch::kF32);
		target_float = target_float.to(torch::kUInt8);

		// Convert data and target from PyTorch Tensor to StdTensor
		auto data = Tensor_to_StdTensor<float, Posit>(data_float);
		auto target = Tensor_to_StdTensor<uint8_t, Target>(target_float);
		
		// Forward pass
		auto output = model.forward(data);
		cross_entropy_loss<Posit> loss(output, target);

		// Backward pass and optimize
		optimizer.zero_grad();
		loss.backward(model);
		optimizer.step();

		// Print progress
		if (++batch_idx % kLogInterval == 0) {
			float loss_value = loss.template item<float>();
			std::printf("Train Epoch: %.3f/%2ld Data: %5ld/%5ld Loss: %.4f\n",
					epoch-1+static_cast<float>(total_batch_size)/dataset_size,
					num_epochs,
					total_batch_size,
					dataset_size,
					loss_value);
		}
	}
}

#endif /* TRAIN_POSIT_HPP */
