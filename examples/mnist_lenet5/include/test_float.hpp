#ifndef TEST_FLOAT_HPP
#define TEST_FLOAT_HPP

// General headers
#include <iostream>
#include <torch/torch.h>

template <class Model, typename DataLoader>
void test_float(	Model& model,
					DataLoader& data_loader,
					size_t dataset_size	){
	
	torch::NoGradGuard no_grad;
	model->eval();

	float test_loss = 0;
	size_t correct = 0;

	for(const auto& batch : data_loader) {
		// Get data and target
		auto data = batch.data;
		auto target = batch.target;
		
		// Convert data and target to float32 and long
		torch::Device device = model->parameters().front().device();
		data = data.to(device, torch::kF32);
		target = target.to(device, torch::kLong);

		// Forward pass
		auto output = model->forward(data);

		// Calculate loss
		test_loss += torch::nll_loss(	output,
						 				target,
						 				/*weight=*/{},
						 				torch::Reduction::Sum	).template item<float>();
	
		auto pred = output.argmax(1);
		correct += pred.eq(target).sum().template item<int64_t>();
	}

	// Get average loss
	test_loss /= dataset_size;

	// Print results
	std::printf("Test set: Loss: %.4f | Accuracy: [%5ld/%5ld] %.4f\n",
	  			test_loss, correct, dataset_size,
	  			static_cast<float>(correct) / dataset_size);
}

#endif /* TEST_FLOAT_HPP */
