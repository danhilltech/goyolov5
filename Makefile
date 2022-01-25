.PHONY: weights

weights:
	rm -rf weights
	DOCKER_BUILDKIT=0 docker build -f utils/Dockerfile.pytorch.amd64.cuda -t danhilltech/goyoloweights:r0.1.1 --progress plain --network=host ./utils
	docker run --rm --runtime nvidia --network host  -v ${PWD}:/var/app --entrypoint /bin/bash "danhilltech/goyoloweights:r0.1.1" /var/app/utils/generate_weights.sh
	
