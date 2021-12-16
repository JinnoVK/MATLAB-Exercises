classdef myknn
    methods(Static)
        
        
        % this function creates a model that will store a copy of the
        % training examples and the corresponding labels, applying
        % standardization to the examples, re-scaling the values so they
        % fall within a comparable range. if z-score standardization is not
        % applied certain large values will dominate the distance
        % calculations and ruin the overall performance of the classifier

        function m = fit(train_examples, train_labels, k)                                                 
            
            % start of standardisation process
            %find the average values in the examples dataset
			m.mean = mean(train_examples{:,:}); 
            %find the standard deviation in the examples dataset
			m.std = std(train_examples{:,:}); 
            for i=1:size(train_examples,1)
                %we subtract the mean values of the original features so
                %that the new ones are more centered on 0, alongisde this
                %the values are then divided by their standard deviation,
                %resulting in data that is scaled properly        
				train_examples{i,:} = train_examples{i,:} - m.mean; 
                train_examples{i,:} = train_examples{i,:} ./ m.std; 
            end
            % end of standardisation process
            
            %store the trained examples in a field of the "m" structure
            m.train_examples = train_examples; 
            %store the trained labels in a field of the "m" structure
            m.train_labels = train_labels; 
            %the final variable of the function is the number of nearest neighbours we want to look for
            m.k = k; 
        
        end
        
        
        %The following function will compute the distance between a single example
        %that needs to be classified, and all the others in the dataset, it
        %will then loop over each one applying standardization and also
        %finding the corresponding class labels. In order to give the
        %actual predicion we compute the mode of the class labels across
        %the given K value

        function predictions = predict(m, test_examples)

            predictions = categorical;
            
            %Here is the loop that looks through each example and applies
            %standardization 

            for i=1:size(test_examples,1)
                
                fprintf('classifying example example %i/%i\n', i, size(test_examples,1));
                
                this_test_example = test_examples{i,:};
                % the standardization being made here is for the exact same
                % reason explained earlier, to make sure our data is
                % re-scaled and centred on 0 stopping unusually large
                % numbers from ruining the classifier
                
                % start of standardisation process
                this_test_example = this_test_example - m.mean;
                this_test_example = this_test_example ./ m.std;
                % end of standardisation process
                
                
                this_prediction = myknn.predict_one(m, this_test_example);
                predictions(end+1) = this_prediction;
            
            end
        
        end
        
        
        %the following function will calculate the distance
        %between examples and store said values in an array called
        %distances
        function prediction = predict_one(m, this_test_example)
            
            distances = myknn.calculate_distances(m, this_test_example);
            neighbour_indices = myknn.find_nn_indices(m, distances);
            prediction = myknn.make_prediction(m, neighbour_indices);
        
        end
        
        % the calculate distances funtion will loop through all the
        % training examples, finding out the dinstance between the current
        % value and the next one by calling from the "calculate_distance"
        % function

        function distances = calculate_distances(m, this_test_example)
            
			distances = [];
            
			for i=1:size(m.train_examples,1)
                
				this_training_example = m.train_examples{i,:};
                this_distance = myknn.calculate_distance(this_training_example, this_test_example);
                distances(end+1) = this_distance;
            end
        
        end
        
        %the calculate distance function finds the euclidian distance
        %between 2 different examples, which will then be called on by
        %other functions
        
        function distance = calculate_distance(p, q)
            
			differences = q - p;
            squares = differences .^ 2;
            total = sum(squares);
            distance = sqrt(total);
        
        end
        
        %this function takes the distances array and sorts it with a second
        %parameter which is the original number of indeces. once this is
        %done it reads out the k indices and returns them to the
        %predict_one function so that we can calculate the nearest
        %neighbours
        

        function neighbour_indices = find_nn_indices(m, distances)
            
			[sorted, indices] = sort(distances);
            neighbour_indices = indices(1:m.k);
        
        end
        
        % find the most common class label amongst the nearest neighbours and store it as our predicted class for this example
        function prediction = make_prediction(m, neighbour_indices)

			neighbour_labels = m.train_labels(neighbour_indices);
            prediction = mode(neighbour_labels);
        
		end

    end
end

