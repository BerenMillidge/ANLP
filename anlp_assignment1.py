
from __future__ import division
import re
import numpy as np
import operator



#This was originally written in Python 3.5.2. I've converted it to Python 2.7, and it seems to work okay. There may be some strange bugs lurking around though, so beware.
# To generate the language model run the build model function, given a file of input data. Once you have the model you can then use it to compute the perplexity or to generate sequences from the model by calling the appropriate functions with the appropriate filenames.

def readFile(fname):    #Gets the lines.
    #Takes a file name, reads the file, and returns a list containing each line of the file.
    with open(fname) as f:
        lines = f.readlines()
        return lines

def preprocess_line(line):
    #This function takes a line, strips it of its line break,converts all numbers to 0, and removes any other characters which are not alphabetic, numbers, or a space or period.
    #It then adds delimiting hashes at the beginning and end of the line, converts all letters to lower case, and then strips the newline character at the end of each line.
    line = line.rstrip()
    line = re.sub('[0-9]', "0",line)
    line = re.sub('[^A-Za-z0. ]',"",line)
    line = "##" + line + "#"
    return line.lower().strip('\n')

def generate_preGram(input, N,pregram,):        #This is the recursive function which generates the pregrams for the N-gram models to compute the marginal sums.
    n = N-1
    if n>-1:
        pregram = pregram + input[n]
        pregram = generate_preGram(input, N-1, pregram,)
    return pregram

def compute_perplexity(model,fname):    #This function computes the perplexity of a language model on the input.
    # It takes in an input file and for each line computes the log probability of the line (the sum of the log probabilities of each character)
    #Then it sums the line log probabilities to find the total log probabilities. Then from there it computes and returns the perplexity.

    lines =readFile(fname)
    perplist = []
    sum3 = 0
    for line in lines:
        sum = 0
        for i in range((len(line)-2)):
            i = i+2
            key = line[i-2] + line[i-1] +line[i]
            if float(model[key])!=0:
                prob = np.log2(float(model[key]))
            else:
                prob = 0
            sum = sum +prob
        perplist.append(sum)
        sum3+=len(line)
    sum2 = 0
    for s in perplist:
        sum2 = sum2 +s
    perplexity = 2 ** -((1/(sum3-2))*sum2)
    print perplexity
    return perplexity

def compute_condProbs(countDict, N):        #The aim of this function is to go through the dictionary of counts, computing the conditionl probability for each N-gram. so P(abc) = P(c|ab).
    #It takes in a dictionary of Ngrams and counts (usually smoothed, though it works without) and returns a language model of Ngrams and their associated conditional probabilities.
    model = {}
    for k,v in countDict.items():
        if len(k) ==N:
            pregram = generate_preGram(k,N-1,"")
            pregram=pregram[::-1]
            sum = 0
            i=0
            for key,value in countDict.items():
                if key[:-1] == pregram:
                    sum = sum+float(value)
                    i = i+1

            if sum >0:
                model[k] = v/sum
            else:
                model[k] = 0
            if N ==3 and pregram[1] =='#' and pregram[0] != '#':
                for key, value in countDict.items():
                    if key[:-1] == pregram:
                        if key !='#':
                            model[k] = 0
                        else:
                            model[k] = 1
    return model

def check_condprobs(model,N):   #This just checks that the sum of all conditional probabilities over its condition = 1, as they should.
    for k,v in model.items():
        pregram = generate_preGram(k,N-1, "")
        pregram = pregram[::-1]
        sum = 0
        for key,value in model.items():
            if key[:-1] == pregram:
                sum = sum+float(value)
        if sum !=1:
            print ("Blimey!")
            print(k)
            print(sum)
    print ("Phew! Done.")


def Ngram_add_alpha_smoothing(count_dict, alpha, N):    #This will do add alpha smoothing for any specified N-gram. To get add-one smoothing, set alpha =1
    smoothed_counts = {}
    for k,v in count_dict.items():
        if N ==1:
            sum = 0
            i = 0
            for key,value in count_dict.items():
                if len(key) ==1:
                    sum = sum + float(value)
                    i = i+1
            smoothedval = (v+alpha/(sum+(alpha*i)))
            smoothed_counts[k] = smoothedval
        if N>1:
            pregram = generate_preGram(k,N-1,"")
            pregram = pregram[::-1]
            sum = 0
            i = 0
            for key,value in count_dict.items():
                if key[:-1] == pregram:
                    sum = sum + float(value)
                    i = i+1
            smoothedval = (v+alpha)/(sum+(alpha*i))
            smoothed_counts[k] = smoothedval
    return smoothed_counts

def countnoncombinatoric(line, N, dict): # this allows you to generate the ngrams without the combinatorial dict initialisation. It returns a dict with only observed ngrams in it.
    for i in range(len(line)):
        ngram = ""
        for j in range(N):
            ngram = ngram + line[i - j]
        ngram = ngram[::-1]  # This reverses the order of the n-grams because I decided to compute it in a silly way!
        if ngram not in dict:
            dict.keys().append(ngram)
            dict[ngram] = 0
        dict[ngram] += 1
        return dict


def countNgram(line, N, dict):  #Counts the Ngrams. It requires an already initialised dictionary to work. It doesn't create new entries when it's seen an unknown.
            for i in range(len(line)):
                ngram = ""
                for j in range(N):
                    ngram = ngram+line[i - j]
                ngram = ngram[::-1]
                dict[ngram] +=1


def recursive_dict(N,recurdict, s):     #The fun bit! This recursively generates new keys for all chars in the valid string, so as to generate all combinations of keys with length N.
    if (N != 0):
        int_dict = {}
        for k in recurdict:
            newdict = {}
            for char in s:
                newdict[k+char] =0
            int_dict.update(newdict)
        recurdict.update(int_dict)
        recursive_dict((N-1), recurdict,s)



def init_dict(N):       #Initialises, then recursively generates all possible N-gram combinations. Sets all values to 0.
    valid_chars = "abcdefghijklmnopqrstuvwxyz0. #"  # Hash here to account for the hash we added at the start of each line to make computing the N-grams easier
    full_dict = {}
    for char in valid_chars:
        full_dict[char] = 0
    recursive_dict(N-1,full_dict,valid_chars)
    return full_dict


def build_model(fileName, NgramN, fname):
# So the aim of this is given a file name and an N-gram N, it computes the language model of the input and then writes it to a file. It computes the counts of N-grams, smoothes them
# then it computes the conditional probabilities, and then writes them to file in the same format roughly as the model we have alerady been given.

    lines = readFile(fileName)
    countDict = init_dict((NgramN))
    for line in lines:
        countNgram(preprocess_line(line), NgramN, countDict)
    culled_dict  = cull_not_Ngrams(countDict, NgramN)
    smoothdict = Ngram_add_alpha_smoothing(culled_dict,1,NgramN)
    model = compute_condProbs(smoothdict,NgramN)
    write_To_file(fname,model)


def createInputFile(filename,inputfile):  #This takes a file we want to test, then preprocesses each line of it to make sure it doesn't contain any characters not included in the trigram model, then writes each processed line to a new file.
    lines = readFile(filename)
    with open(inputfile, 'w') as f:
        for line in lines:
            line = preprocess_line(line)
            f.write(line)


def write_To_file(fname, model):   #This writes our language model to a file. The file is formatted such that each line contains a trigram and the associated conditional probability, separated by a tab.
    keylist = sorted(model.keys())
    keylist.sort()
    with open(fname, "w") as f:
        for key in keylist:
            f.write("%s" '\t' "%s" '\n' % (key,model[key])) # This puts it into the same format as their model. This means we can read in both files with the same function.


def cull_not_Ngrams(model,N):  #This makes sure that the dictionary contains only the correct N-gram type you want. This is probably a really inefficient way to do it, but most compute time is spent on the smoothing anyway.
    #It takes a model and an Ngram N and makes sure that only Ngrams with that N remain in the model. It's basically doing cleanup in case anything weird got in there by mistake.
    new_model = {}
    for k,v in model.items():
        if len(k) == N:
            new_model[k]=v
    return new_model

def read_in_model(fname):   #This takes in a file written in the required format and generates a dictionary from it which has the trigrams as its keys and their associated probabilities as its values. This dictionary can then be used to generate sequences or compute perplexities or similar.
    with open(fname) as f:
        file = f.readlines()
    model = {}
    for f in file:
        s = f.split("\t")
        model[s[0]] = s[1].strip('\n')
    return model

def generate_from_lm(model,N):  #This function takes a language model and uses it to generate a sequence of characters governed by the probabilities in the language model. It starts with an initial string '##' which signifies the beginning of the line,
    #then generates new characters using the probabilities of the trigrams stored in the model. Once the string is generated, it then converts all double #s to newlines.
    valid_chars = "abcdefghijklmnopqrstuvwxyz0. #"
    s = '##'
    for i in range(N-2):
        pregram = s[i] + s[i+1]
        distribution = {}
        for k,v in model.items():
            if k[0] ==pregram[0] and k[1] == pregram[1]:
                distribution[k[2]] = float(v)
        if len(distribution) ==0:
            for char in valid_chars:
                distribution[i] = 1/len(valid_chars)
        l =sum(distribution.values())
        diff = 1-l
        probabilities_sorted = sorted(distribution.items(), key=operator.itemgetter(1))
        probabilities_sorted_reversed = sorted(distribution.items(), key=operator.itemgetter(1), reverse=True)
        if diff > 0:
            key_lowest = probabilities_sorted[0][0]
            distribution[key_lowest] += diff
        else:
            key_highest = probabilities_sorted_reversed[0][0]
            distribution[key_highest] += diff
        outcomes = []
        probs = []
        for k,v in distribution.items():
            outcomes.append(k)
            probs.append(v)
        outcomes = np.array(outcomes)
        probs = np.array(probs)
        bins = []
        sum_of_probabilities = 0
        j = 0
        for prob in probs:
            sum_of_probabilities +=prob
            bins.append(sum_of_probabilities)
            j = j+1
        char = outcomes[np.digitize(np.random.random_sample(1), bins)]
        #print char[0]
        if pregram ==".#":
            pregram = ". "
            s = s + s[i-3]  #We need to get a serious actual fix for this. I'm not sure how it's even generating the .# pregrams!
            #I think the problem is that it can generate .# as chars but there is no pregram containing that in the model, so it breaks
            #For some reason when this happens our 'char' becomes a number, and so the whole thing breaks, but I'm not sure why it does that!
        else:
            s = s + char[0]
    new_s = ""
    for i in range (N): #Adds a newline whenever there is a '##'.
        if s[i] == '#' and s[i+1] == '#':
            new_s = new_s + '\n'
            i = i+1
        new_s = new_s + s[i]
    print(new_s)






# Below is the area where we call our functions. To build a language moel and write it to a file, call the build model_model function with the training document, the N-gram count you want, and the file you want it to write the model to.
#To read in a model, use the read_in_model function which takes the filename you want it to read from.
#To generate a sequence from the model, use the generate_from_model function which takes the model you want to use and the number of characters you wwant to generate.
#To compute the perplexity of an input string under a specific language model, use the compute_perplexity functoin which takes in a model, and the file name of the test data.

#build_model("/afs/inf.ed.ac.uk/user/s16/s1686853/anlp_assignment1/training.en",3, "/afs/inf.ed.ac.uk/user/s16/s1686853/anlp_assignment1/trigramModelEnglish")

#model_english = read_in_model("/afs/inf.ed.ac.uk/user/s16/s1686853/anlp_assignment1/trigramModelEnglish")
#createInputFile("/afs/inf.ed.ac.uk/user/s16/s1686853/anlp_assignment1/test", "/afs/inf.ed.ac.uk/user/s16/s1686853/anlp_assignment1/testinput")
#print (compute_perplexity(model_english,"/afs/inf.ed.ac.uk/user/s16/s1686853/anlp_assignment1/testinput"))
#print generate_from_lm(model_english,300)
#model_theirs = read_in_model("/afs/inf.ed.ac.uk/user/s16/s1686853/anlp_assignment1/model-br.en")
#print generate_from_lm(model_theirs, 300)
#model_german = read_in_model("D:/beren/Documents/Edinburgh/assignment/TrigramModelGerman.txt")
#model_spanish = read_in_model("D:/beren/Documents/Edinburgh/assignment/TrigramModelSpanish.txt")
#model_theirs = read_in_model("D:/beren/Documents/Edinburgh/assignment/model-bren.txt")

#model_english3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/TrigramModelEnglish3.txt")
#model_german3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/TrigramModelGerman3.txt")
#model_spanish3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/TrigramModelSpanish3.txt")
#model_english5 = read_in_model("D:/beren/Documents/Edinburgh/assignment/TrigramModelEnglish5.txt")

#model_bigram_english3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/BigramModelEnglish3.txt")
#model_bigram_german3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/BigramModelGerman3.txt")
#model_bigram_spanish3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/BigramModelSpanish3.txt")

#model_unigram_english3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/UnigramModelEnglish3.txt")
#model_unigram_german3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/UnigramModelGerman3.txt")
#model_unigram_spanish3 = read_in_model("D:/beren/Documents/Edinburgh/assignment/UnigramModelSpanish3.txt")

#model_english4 = read_in_model("D:/beren/Documents/Edinburgh/assignment/TrigramModelEnglish4.txt")
#print(compute_perplexity(model_english4,"D:/beren/Documents/Edinburgh/assignment/testinput.txt"))
#check_condprobs(model_english4,3)


#generate_from_lm(model_english5,300)
#generate_from_lm(model_theirs,300)


#test_model = {"##a" : 0.2, "#aa" :0.2, "#ba" : 0.15, "aaa" : 0.4, "aba" : 0.6, "baa" : 0.25, "bba" : 0.5,
#"##b" : 0.8, "#ab" : 0.7, "#bb" : 0.75, "aab" : 0.5, "abb" : 0.3, "bab" : 0.65, "bbb" : 0.4,
#"###" : 0.0, "#a#" : 0.1, "#b#" : 0.1, "aa#" : 0.1, "ab#" : 0.1, "ba#" : 0.1, "bb#" : 0.1}

#createInputFile("D:/beren/Documents/Edinburgh/assignment/ProperTest.txt","D:/beren/Documents/Edinburgh/assignment/TestInput.txt")

#print(compute_perplexity(model_english5,"D:/beren/Documents/Edinburgh/assignment/TestInput.txt"))

#generate_from_lm(model_english5,300)
#generate_from_lm(model_theirs,300)
#print(compute_perplexity(test_model,"D:/beren/Documents/Edinburgh/assignment/test.txt"))
#print(compute_perplexity(model_english2,"D:/beren/Documents/Edinburgh/assignment/testinput.txt"))
#print(compute_perplexity(model_english,"D:/beren/Documents/Edinburgh/assignment/EnglishTraining.txt"))

#print(compute_perplexity(model_english3,"D:/beren/Documents/Edinburgh/assignment/testinput.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/testinput.txt"))
#print(compute_perplexity(model_spanish3,"D:/beren/Documents/Edinburgh/assignment/testinput.txt"))

#print(compute_perplexity(model_english,"D:/beren/Documents/Edinburgh/assignment/germantestinput.txt"))
#print(compute_perplexity(model_english,"D:/beren/Documents/Edinburgh/assignment/testinput.txt"))
#print(compute_perplexity(model_english,"D:/beren/Documents/Edinburgh/assignment/spanishtestinput.txt"))



#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindEnglish.txt","D:/beren/Documents/Edinburgh/assignment/InputEnglish.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindGerman.txt","D:/beren/Documents/Edinburgh/assignment/InputGerman.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindSpanish.txt","D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindTagalog.txt","D:/beren/Documents/Edinburgh/assignment/InputTagalog.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindHebrew.txt","D:/beren/Documents/Edinburgh/assignment/InputHebrew.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindBasque.txt","D:/beren/Documents/Edinburgh/assignment/InputBasque.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindRussian.txt","D:/beren/Documents/Edinburgh/assignment/InputRussian.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindUyghur.txt","D:/beren/Documents/Edinburgh/assignment/InputUyghur.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/InputMandarin.txt","D:/beren/Documents/Edinburgh/assignment/InputMandarin.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindFrench.txt","D:/beren/Documents/Edinburgh/assignment/InputFrench.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindLatin.txt","D:/beren/Documents/Edinburgh/assignment/InputLatin.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindPortuguese.txt","D:/beren/Documents/Edinburgh/assignment/InputPortuguese.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindKurdish.txt","D:/beren/Documents/Edinburgh/assignment/InputKurdish.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindSwedish.txt","D:/beren/Documents/Edinburgh/assignment/InputSwedish.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindPolish.txt","D:/beren/Documents/Edinburgh/assignment/InputPolish.txt")
#createInputFile("D:/beren/Documents/Edinburgh/assignment/NorthWindJapanese.txt","D:/beren/Documents/Edinburgh/assignment/InputJapanese.txt")

#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputEnglish.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputGerman.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputFrench.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputLatin.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputPortuguese.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputSwedish.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputPolish.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputRussian.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputTagalog.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputHebrew.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputBasque.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputUyghur.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputMandarin.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputKurdish.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputJapanese.txt"))



#print(compute_perplexity(model_english3,"D:/beren/Documents/Edinburgh/assignment/InputGerman.txt"))
#print(compute_perplexity(model_german3,"D:/beren/Documents/Edinburgh/assignment/InputGerman.txt"))
#print(compute_perplexity(model_spanish3,"D:/beren/Documents/Edinburgh/assignment/InputGerman.txt"))


#print("\n")

#print(compute_perplexity(model_bigram_english,"D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt"))
#print(compute_perplexity(model_bigram_german,"D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt"))
#print(compute_perplexity(model_bigram_spanish,"D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt"))

#print('\n')


#print(compute_perplexity(model_unigram_english,"D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt"))
#print(compute_perplexity(model_unigram_german,"D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt"))
#print(compute_perplexity(model_unigram_spanish,"D:/beren/Documents/Edinburgh/assignment/InputSpanish.txt"))


