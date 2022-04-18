# Import modules
import os

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"
import json

#import matplotlib.pyplot as plt

###########################################################
# CONSTANTS
# For use as dictionary keys in training/testing sets and sums
# DONE - Do not modify.
###########################################################

POP_DENSITY = "Population-Density"
PERCENT_OVR_65 = "Percent-Over-65"
INCOME = "Income"
PERCENT_COLLEGE = "Percent-Attend-College"
VAX_RATE = "Vax-Rate"
VAX_PERC_LIM = 55
SVI = "Social Vulnerability Index"

CONTINUOUS = [POP_DENSITY, PERCENT_OVR_65, INCOME, PERCENT_COLLEGE]
CATEGORICAL = [SVI]
ATTRIBUTES = [POP_DENSITY, PERCENT_OVR_65, INCOME, PERCENT_COLLEGE, SVI]
ALL_KEYS = [
    POP_DENSITY, PERCENT_OVR_65, INCOME, PERCENT_COLLEGE, SVI, VAX_RATE
]


def load_data(filename):
    ''' DO NOT TOUCH. This is already completed for you.
    Loads the data into the program.

    - Reads in JSON file with a dictionary

    Parameters:
    - filename: name of the data file with the training data records

    Returns: 
    data_set: a dictionary of training records with
    - keys, integers, that represent US counties or equivalents
    using FIPS (Federal Information Processing Standards)
    - values, a dictionary with key-value pairs providing information
    on each county. The keys, strings, identify an attribute and the
    values are the numerical value, floats, or a string.

  '''
    with open(filename) as json_file:
        data_set = json.load(json_file)
    return data_set


def organize_data(training_set):
    '''
  Sorts data for highly- and lowly- vaccinated populations
    
  Parameters:
    training_set: a dictionary
    
  Returns:
    two dictionaries: low_vax and high_vax
    The keys for both dictionaries are the strings 
      in the list ATTRIBUTES. 
    The values are lists.
      Lists of floats for the CONTINUOUS attributes.
      Lists of strings for the CATEGORICAL attribute.
  '''
    # Initialize dictionaries to aggregate data
    low_vax = {}
    high_vax = {}

    # Add code here
    for county in training_set:

        if training_set[county]["Vax-Rate"] > 55:
            for attribute in ATTRIBUTES:
                if attribute not in high_vax:
                    high_vax[attribute] = [training_set[county][attribute]]
                else:
                    high_vax[attribute] += [training_set[county][attribute]]
        else:
            for attribute in ATTRIBUTES:
                if attribute not in low_vax:
                    low_vax[attribute] = [training_set[county][attribute]]
                else:
                    low_vax[attribute] += [training_set[county][attribute]]

    return low_vax, high_vax


def train_classifier(low_vax, high_vax):
    '''
  Create the model with average values for each of the ATTRIBUTES
    
  Parameters:
  two dictionaries, low_vax and high_ vax: each with
  -keys, strings, representing each of the county attributes
  -values
    lists of floats for CONTINUOUS attributes
    lists of strings for CATEGORICAL attributes          
        
  Returns:
  two dictionaries: model_low and model_high
  Each dictionary has the 
  - keys, strings, representing the county attributes
  - values
    For CONTINUOUS attributes: floats, the average 
      of all values of an attribute for all counties
    For CATEGORICAL attributes: a sub-dictionary 
      with keys of strings and 
      values of fractions, the proportion of each string
        present in low_vax or high_vax
  '''

    # Initialize the dictionaries
    model_low = {}
    model_high = {}

    # Add code here
    for attribute in low_vax:
        if attribute in CONTINUOUS:
            total = sum(low_vax[attribute])
            avg = total / (len(low_vax[attribute]))
            model_low[attribute] = avg
        else:
            letters = low_vax[attribute]
            total = len(letters)
            svi = {}
            for letter in letters:
                if letter not in svi:
                    svi[letter] = 1 / total
                else:
                    svi[letter] += 1 / total
            model_low[attribute] = svi

    for attribute in high_vax:
        if attribute in CONTINUOUS:
            total = sum(high_vax[attribute])
            avg = total / (len(high_vax[attribute]))
            model_high[attribute] = avg
        else:
            letters = high_vax[attribute]
            total = len(letters)
            svi = {}
            for letter in letters:
                if letter not in svi:
                    svi[letter] = 1 / total
                else:
                    svi[letter] += 1 / total
            model_high[attribute] = svi

    return model_low, model_high


def classify_test_records(test_set, model_low, model_high):
    '''
  Classifies a test set of counties as 
  having a low or high vaccination percentage
   
  Parameters:
  -test_set, a dictionary,
  -model_low, a dictionary,
  -model_high, a dictionary,
        
  Return:
  -test_set, the same as the input dictionary 
  with an added key-value pair:
    Key to be added: "Predicted Vax"
    Value to be added: a string, 
      ">"+str(VAX_PERC_LIM)+"%" if the model predicts a vaccination
        rate above the VAX_PERC_LIM
      "<"+str(VAX_PERC_LIM)+"%" if the model predicts a vaccination
        rate below the VAX_PERC_LIM
      "Unknown" if the model cannot 
        predict high or low vaccination rates  
  '''

    for county in test_set:
        high_count = 0
        low_count = 0
        for attribute in ATTRIBUTES:
            if attribute in CONTINUOUS:
                low_diff = model_low[attribute] - test_set[county][attribute]
                low_diff = abs(low_diff)
                high_diff = model_high[attribute] - test_set[county][attribute]
                high_diff = abs(high_diff)
                if low_diff < high_diff:
                    low_count += 1
                else:
                    high_count += 1
            else:
                letter = test_set[county][attribute]

                if letter in model_low[attribute] and letter in model_high[
                        attribute]:

                    if model_low[attribute][letter] < model_high[attribute][
                            letter]:
                        high_count += 1

                    else:
                        low_count += 1

                elif letter in model_low[
                        attribute] and letter not in model_high[attribute]:

                    low_count += 1
                elif letter not in model_low[
                        attribute] and letter in model_high[attribute]:
                    high_count += 1
                else:
                    pass

        if high_count < low_count:
            test_set[county]["Predicted Vax"] = "<=" + str(VAX_PERC_LIM) + "%"
        elif high_count > low_count:
            test_set[county]["Predicted Vax"] = ">" + str(VAX_PERC_LIM) + "%"
        else:
            test_set[county]["Predicted Vax"] = "Unknown"

    return test_set


def determine_accuracy(test_set):
    ''' 
  Determines the accuracy of the model.

  Parameters:
  - test_set, a dictionary

  Returns: 
  - num_correct, num_incorrect, accuracy
  The results are defined as:
    - num_correct, an integer, number of records correctly classified
    - num_incorrect, an integer, number of records incorrectly classified
    - accuracy, a float, the number correct / total number of records
      the fraction correctly predicted           
  '''
    num_correct = 0
    num_incorrect = 0
    total = len(test_set)
    for county in test_set:
        if test_set[county]["Predicted Vax"] == "<=" + str(VAX_PERC_LIM) + "%":
            actual = test_set[county]['Vax-Rate']
            if actual <= VAX_PERC_LIM:
                num_correct += 1
            else:
                num_incorrect += 1
        elif test_set[county]["Predicted Vax"] == ">" + str(
                VAX_PERC_LIM) + "%":
            actual = test_set[county]['Vax-Rate']
            if actual > VAX_PERC_LIM:
                num_correct += 1
            else:
                num_incorrect += 1
        else:
            num_incorrect += 1
    accuracy = num_correct / total
    return num_correct, num_incorrect, accuracy


def report_accuracy(num_correct, num_incorrect, accuracy):
    '''
  Print 
  - the number of correct and incorrect predictions 
  - the overall accuracy
 
  Parameters:
  - num_correct, num_incorrect, accuracy
  The results are defined as:
    - num_correct, an integer, number of records correctly classified
    - num_incorrect, an integer, number of records incorrectly classified
    - accuracy, a float, the number correct / total number of records
      the fraction correctly predicted           
 
  Returns: None
  '''
    header = '-' * 16
    print(f'\n{header}')
    print("Results".center(16))
    print(header)
    print(f'Number Correct: {num_correct}')
    print(f'Number Incorrect: {num_incorrect}')
    print(f'Accuracy: {accuracy}\n')


def sensitivity_analysis(test_set, model_low, model_high):
    '''
  Return a dictionary of accuracies for each attribute in ATTRIBUTES
  
  Parameters:
  -  test_set, a dictionary, data for different counties
  -  model_low, a dictionary, 
  -  model_high, a dictionary
 
  Returns:
  - results, a dictionary of key-value pairs
          the keys are the attributes in ATTRIBUTES
          the values are the accuracy due to a single attribute
  '''
    results = {}
    for key in ATTRIBUTES:
        if key in CONTINUOUS:
            test_set = classify_test_records_continuous(
                test_set, model_low, model_high, key)
            num_correct, num_incorrect, accuracy = determine_accuracy(test_set)
            results[key] = accuracy

        else:
            test_set = classify_test_records_categorical(
                test_set, model_low, model_high, key)
            num_correct, num_incorrect, accuracy = determine_accuracy(test_set)
            results[key] = accuracy
    return results

    # Add code here

    return results


def report_sensitivity(results):
    ''' 
  Prints the accuracies determined in the sensitivity analysis
    
  Parameter:
  - results, a dictionary with attributes as keys 
    and accuracies as values
    
  Returns: None
  '''
    header = 54 * "-"
    print('\n')
    print("Sensitivity Analysis".center(54))
    print(header)
    print(f'{"Attribute":^25} | {"Accuracy":^26}')
    print(header)
    for attribute in results:
        print(f'{attribute:^26}{str(round(results[attribute],2)).center(32)}')
    print('\n')


def visualize_sensitivity(results):
    '''
  Displays results of the sensitivity analysis
  in graphical form
  
  Parameter:
  - results, a dictionary with attributes as keys 
    and accuracies as values
  '''
    import matplotlib.pyplot as plt
    attributes = list(results.keys())
    values = list(results.values())
    labels = attributes
    x = range(len(attributes))
    plt.xticks(x, labels, rotation='vertical')
    plt.xlabel("Attribute")
    plt.ylabel("Accuracy")
    plt.title("Sensitivity Analysis")
    plt.bar(attributes, values)
    plt.savefig('Sensitivity Analysis Visualization.pdf', )


###########################################################
# Helper functions
###########################################################


def dprint(my_dict):
    ''' A helper function that '''
    print(json.dumps(my_dict, indent=1))


def classify_test_records_continuous(test_set, model_low, model_high, key):
    '''
  Classifies using one attribute a test set of counties as 
  having a low or high vaccination percentage
   
  Parameters:
  -test_set, a dictionary,
  -model_low, a dictionary,
  -model_high, a dictionary,
  -key, a string, a continuous attribute
        
  Return:
  -test_set, the same as the input dictionary 
  with an added key-value pair:
    Key to be added: "Predicted Vax"
    Value to be added: a string, 
      ">"+str(VAX_PERC_LIM)+"%" if the model predicts a vaccination
        rate above the VAX_PERC_LIM
      "<="+str(VAX_PERC_LIM)+"%" if the model predicts a vaccination
        rate below or equal to the VAX_PERC_LIM
      "Unknown" if the model cannot 
        predict high or low vaccination rates  
  '''
    for county in test_set:
        low_diff = model_low[key] - test_set[county][key]
        low_diff = abs(low_diff)
        high_diff = model_high[key] - test_set[county][key]
        high_diff = abs(high_diff)
        if low_diff < high_diff:
            test_set[county]["Predicted Vax"] = "<=" + str(VAX_PERC_LIM) + "%"
        elif low_diff > high_diff:
            test_set[county]["Predicted Vax"] = ">" + str(VAX_PERC_LIM) + "%"
        else:
            test_set[county]["Predicted Vax"] = "Unknown"
    return test_set


def classify_test_records_categorical(test_set, model_low, model_high, key):
    '''
  Classifies a test set of counties as 
  having a low or high vaccination percentage
   
  Parameters:
  -test_set, a dictionary,
  -model_low, a dictionary,
  -model_high, a dictionary,
  -key, a string, a categorical attribute
        
  Return:
  -test_set, the same as the input dictionary 
  with an added key-value pair:
    Key to be added: "Predicted Vax"
    Value to be added: a string, 
      ">"+str(VAX_PERC_LIM)+"%" if the model predicts a vaccination
        rate above the VAX_PERC_LIM
      "<="+str(VAX_PERC_LIM)+"%" if the model predicts a vaccination
        rate below or equal to the VAX_PERC_LIM
      "Unknown" if the model cannot 
        predict high or low vaccination rates  
  '''
    for county in test_set:
        letter = test_set[county][key]

        if letter in model_low[key] and letter in model_high[key]:

            if model_low[key][letter] < model_high[key][letter]:
                test_set[county]["Predicted Vax"] = ">" + str( VAX_PERC_LIM) + "%"

            elif model_low[key][letter] > model_high[key][letter]:
                test_set[county]["Predicted Vax"] = "<=" + str(VAX_PERC_LIM) + "%"
            else:
                test_set[county]["Predicted Vax"] = "Unknown"

        elif letter in model_low[key] and letter not in model_high[key]:

            test_set[county]["Predicted Vax"] = "<=" + str(VAX_PERC_LIM) + "%"
        elif letter not in model_low[key] and letter in model_high[key]:
            test_set[county]["Predicted Vax"] = ">" + str(VAX_PERC_LIM) + "%"
        else:
            test_set[county]["Predicted Vax"] = "Unknown"
    return test_set


###########################################################
# main - starts the program
###########################################################
def main():

    print("Reading in training data...")
    filename = "bigger_training.json"

    training_set = load_data(filename)
    print("Done reading training data.\n")
    #dprint(training_set)

    # ------------- PART 1 ------------------#
    print("Organizing the data...")
    # TODO: Organize the data
    low_vax, high_vax = organize_data(training_set)

    print("Creating the model...")
    # TODO: Create the model.
    model_low, model_high = train_classifier(low_vax, high_vax)
    #print(model_low)
    #print(model_high)
    #Prepare test_set (Done for you)
    print("Reading in test data...")
    filename = "bigger_test_set.json"
    #filename = "bigger_test_set.json"
    test_set = load_data(filename)
    print("Done reading training data.\n")

    print("Classifying records...")
    # TODO: Complete train_classifier.
    test_set = classify_test_records(test_set, model_low, model_high)

    print("Done classifying.\n")

    # ------------- PART 2 ------------------#
    print("Determining accuracy...")
    #  TODO: Determine the accuracy of the model.
    num_correct, num_incorrect, accuracy = determine_accuracy(test_set)

    print("Printing results...")
    #  TODO: Print results.
    report_accuracy(num_correct, num_incorrect, accuracy)

    print("Presenting visualization...")

    results = sensitivity_analysis(test_set, model_low, model_high)
    report_sensitivity(results)
    visualize_sensitivity(results)

    print("Program finished.")


if __name__ == "__main__":
    main()
