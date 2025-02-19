import spacy
from spacy.training.example import Example
import random


TRAIN_DATA = [
    
    ("Beetles are fascinating insects with unique adaptations.", {"entities": [(0, 6, "BEETLES")]}),
    ("Do beetles have wings?", {"entities": [(3, 9, "BEETLES")]}),
    ("Beetles can be found in almost every habitat.", {"entities": [(0, 6, "BEETLES")]}),
    ("Some beetles produce light through bioluminescence.", {"entities": [(5, 11, "BEETLES")]}),
    ("Beetles play a crucial role in ecosystems as decomposers.", {"entities": [(0, 6, "BEETLES")]}),
    ("Certain beetles have powerful mandibles for defense.", {"entities": [(8, 14, "BEETLES")]}),
    ("Beetles have a hard exoskeleton for protection.", {"entities": [(0, 6, "BEETLES")]}),
    ("Some beetles mimic other insects to avoid predators.", {"entities": [(5, 11, "BEETLES")]}),
    ("Beetles can be herbivorous, carnivorous, or omnivorous.", {"entities": [(0, 6, "BEETLES")]}),
    ("There are over 350,000 species of beetles worldwide.", {"entities": [(31, 37, "BEETLES")]}),
    


    ("Beetles are fascinating insects with unique adaptations.", {"entities": [(0, 6, "BEETLES")]}),
    ("Do beetles have wings?", {"entities": [(3, 9, "BEETLES")]}),
    ("Beetles can be found in almost every habitat.", {"entities": [(0, 6, "BEETLES")]}),
    ("Some beetles produce light through bioluminescence.", {"entities": [(5, 11, "BEETLES")]}),
    ("Beetles play a crucial role in ecosystems as decomposers.", {"entities": [(0, 6, "BEETLES")]}),
    ("Certain beetles have powerful mandibles for defense.", {"entities": [(8, 14, "BEETLES")]}),
    ("Beetles have a hard exoskeleton for protection.", {"entities": [(0, 6, "BEETLES")]}),
    ("Some beetles mimic other insects to avoid predators.", {"entities": [(5, 11, "BEETLES")]}),
    ("Beetles can be herbivorous, carnivorous, or omnivorous.", {"entities": [(0, 6, "BEETLES")]}),
    ("There are over 350,000 species of beetles worldwide.", {"entities": [(31, 37, "BEETLES")]}),
    ("Beetles thrive in tropical and temperate climates.", {"entities": [(0, 6, "BEETLES")]}),
    ("Some beetles are beneficial as pollinators.", {"entities": [(5, 11, "BEETLES")]}),
    ("Certain beetles can produce loud sounds to deter predators.", {"entities": [(8, 14, "BEETLES")]}),
    ("Butterflies have colorful wings and delicate bodies.", {"entities": [(0, 9, "BUTTERFLY")]}),
    ("Do butterflies migrate long distances?", {"entities": [(3, 12, "BUTTERFLY")]}),
    ("Butterflies are important pollinators in many ecosystems.", {"entities": [(0, 9, "BUTTERFLY")]}),
    ("Some butterflies can camouflage to avoid predators.", {"entities": [(5, 14, "BUTTERFLY")]}),
    ("Butterflies undergo metamorphosis from caterpillars.", {"entities": [(0, 9, "BUTTERFLY")]}),
    ("Certain butterflies have striking patterns to deter threats.", {"entities": [(8, 17, "BUTTERFLY")]}),
    ("Butterflies feed on nectar from flowers.", {"entities": [(0, 9, "BUTTERFLY")]}),
    ("Some butterflies have a short lifespan of only a few weeks.", {"entities": [(5, 14, "BUTTERFLY")]}),
    ("Butterflies are admired for their beauty worldwide.", {"entities": [(0, 9, "BUTTERFLY")]}),
    ("There are over 17,000 species of butterflies globally.", {"entities": [(31, 40, "BUTTERFLY")]}),
    ("Butterflies use their antennae to sense the environment.", {"entities": [(0, 9, "BUTTERFLY")]}),
    ("Some butterflies mimic toxic species for protection.", {"entities": [(5, 14, "BUTTERFLY")]}),
    ("Butterflies taste with their feet.", {"entities": [(0, 9, "BUTTERFLY")]}),
    ("Butterflies can travel thousands of miles during migration.", {"entities": [(0, 9, "BUTTERFLY")]}),

    ("Cats are independent animals that love to explore.", {"entities": [(0, 4, "CAT")]}),
    ("Do cats like to climb trees?", {"entities": [(3, 7, "CAT")]}),
    ("Cats have sharp claws and excellent night vision.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats enjoy playing with toys and chasing objects.", {"entities": [(5, 9, "CAT")]}),
    ("Cats purr when they feel comfortable and happy.", {"entities": [(0, 4, "CAT")]}),
    ("Certain cats have unique fur patterns and colors.", {"entities": [(8, 12, "CAT")]}),
    ("Cats are known for their agility and flexibility.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats prefer solitude while others seek attention.", {"entities": [(5, 9, "CAT")]}),
    ("Cats can jump up to six times their body length.", {"entities": [(0, 4, "CAT")]}),
    ("There are many different breeds of cats worldwide.", {"entities": [(35, 39, "CAT")]}),
    ("Cats are known for their independence and agility.", {"entities": [(0, 4, "CAT")]}),
    ("Do cats sleep for most of the day?", {"entities": [(3, 7, "CAT")]}),
    ("Cats have retractable claws to help them hunt.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats prefer to live indoors, while others roam outside.", {"entities": [(5, 9, "CAT")]}),
    ("Cats communicate using body language and vocalizations.", {"entities": [(0, 4, "CAT")]}),
    ("Certain cats have unique fur colors and patterns.", {"entities": [(8, 12, "CAT")]}),
    ("Cats are often found grooming themselves throughout the day.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats are more social, while others prefer solitude.", {"entities": [(5, 9, "CAT")]}),
    ("Cats can jump up to six times their body length.", {"entities": [(0, 4, "CAT")]}),
    ("There are many different breeds of cats worldwide.", {"entities": [(35, 39, "CAT")]}),
    ("Cats purr when they feel content and relaxed.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats love to chase laser pointers.", {"entities": [(5, 9, "CAT")]}),
    ("Cats have excellent night vision and can see in low light.", {"entities": [(0, 4, "CAT")]}),
    ("Certain cats develop strong bonds with their owners.", {"entities": [(8, 12, "CAT")]}),
    ("Cats use their whiskers to detect changes in their environment.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats enjoy playing with interactive toys.", {"entities": [(5, 9, "CAT")]}),
    ("Cats are curious creatures that love exploring their surroundings.", {"entities": [(0, 4, "CAT")]}),
    ("There are wild cat species such as lions and tigers.", {"entities": [(11, 14, "CAT")]}),
    ("Cats have a highly developed sense of smell.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats can recognize their names and respond to them.", {"entities": [(5, 9, "CAT")]}),
    ("Cats use their tails for balance and communication.", {"entities": [(0, 4, "CAT")]}),
    ("Certain cats are more vocal and meow frequently.", {"entities": [(8, 12, "CAT")]}),
    ("Cats love finding cozy spots to nap in the house.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats form strong attachments to their owners.", {"entities": [(5, 9, "CAT")]}),
    ("Cats can squeeze through surprisingly small gaps.", {"entities": [(0, 4, "CAT")]}),
    ("There are cats that prefer human companionship over solitude.", {"entities": [(11, 15, "CAT")]}),
    ("Cats have sharp teeth adapted for eating meat.", {"entities": [(0, 4, "CAT")]}),
    ("Some cats enjoy climbing high places and observing their surroundings.", {"entities": [(5, 9, "CAT")]}),
    ("Cats are natural hunters and love chasing moving objects.", {"entities": [(0, 4, "CAT")]}),
    ("Certain cats have long fur that requires regular grooming.", {"entities": [(8, 12, "CAT")]}),

    ("Cows are domesticated animals raised for milk and meat.", {"entities": [(0, 4, "COW")]}),
    ("Do cows recognize their owners?", {"entities": [(3, 7, "COW")]}),
    ("Cows have a strong sense of smell and hearing.", {"entities": [(0, 4, "COW")]}),
    ("Some cows have distinct coat patterns.", {"entities": [(5, 9, "COW")]}),
    ("Cows communicate through vocalizations and body language.", {"entities": [(0, 4, "COW")]}),
    ("Certain cows produce more milk than others.", {"entities": [(8, 12, "COW")]}),
    ("Cows are social animals and prefer to stay in groups.", {"entities": [(0, 4, "COW")]}),
    ("Some cows are used in farming for plowing fields.", {"entities": [(5, 9, "COW")]}),
    ("Cows have four stomach compartments for digestion.", {"entities": [(0, 4, "COW")]}),
    ("There are many breeds of cows worldwide.", {"entities": [(27, 31, "COW")]}),
    ("Cows are domesticated animals often raised for milk and meat.", {"entities": [(0, 4, "COW")]}),
    ("Do cows recognize and respond to their names?", {"entities": [(3, 7, "COW")]}),
    ("Cows have four stomach compartments to aid digestion.", {"entities": [(0, 4, "COW")]}),
    ("Some cows have distinctive coat colors and patterns.", {"entities": [(5, 9, "COW")]}),
    ("Cows are social animals and prefer to live in herds.", {"entities": [(0, 4, "COW")]}),
    ("Certain cows produce higher amounts of milk than others.", {"entities": [(8, 12, "COW")]}),
    ("Cows communicate using various vocalizations and body movements.", {"entities": [(0, 4, "COW")]}),
    ("Some cows are used in farming for plowing fields.", {"entities": [(5, 9, "COW")]}),
    ("Cows have a keen sense of smell and can detect scents from miles away.", {"entities": [(0, 4, "COW")]}),
    ("There are many different breeds of cows worldwide.", {"entities": [(35, 39, "COW")]}),
    ("Cows spend most of their day grazing on grass.", {"entities": [(0, 4, "COW")]}),
    ("Some cows form strong bonds with their caretakers.", {"entities": [(5, 9, "COW")]}),
    ("Cows have wide, flat teeth adapted for grinding plants.", {"entities": [(0, 4, "COW")]}),
    ("Certain cows are raised primarily for beef production.", {"entities": [(8, 12, "COW")]}),
    ("Cows can live for more than 20 years in good conditions.", {"entities": [(0, 4, "COW")]}),
    ("Some cows prefer open pastures, while others adapt to smaller farms.", {"entities": [(5, 9, "COW")]}),
    ("Cows are gentle creatures that can recognize human faces.", {"entities": [(0, 4, "COW")]}),
    ("There are over 800 different breeds of cows in the world.", {"entities": [(33, 37, "COW")]}),

    ("Dogs are loyal companions and great pets.", {"entities": [(0, 4, "DOG")]}),
    ("Do dogs bark at strangers?", {"entities": [(3, 7, "DOG")]}),
    ("Dogs have an exceptional sense of smell.", {"entities": [(0, 4, "DOG")]}),
    ("Some dogs are trained to assist people with disabilities.", {"entities": [(5, 9, "DOG")]}),
    ("Dogs communicate through barking, whining, and body language.", {"entities": [(0, 4, "DOG")]}),
    ("Certain dogs excel in search and rescue missions.", {"entities": [(8, 12, "DOG")]}),
    ("Dogs require regular exercise and a balanced diet.", {"entities": [(0, 4, "DOG")]}),
    ("Some dogs are highly intelligent and easy to train.", {"entities": [(5, 9, "DOG")]}),
    ("Dogs have been companions to humans for thousands of years.", {"entities": [(0, 4, "DOG")]}),
    ("There are many different breeds of dogs worldwide.", {"entities": [(35, 39, "DOG")]}),

    ("Elephants are the largest land animals on Earth.", {"entities": [(0, 8, "ELEPHANT")]}),
    ("Do elephants have excellent memory?", {"entities": [(3, 11, "ELEPHANT")]}),
    ("Elephants use their trunks for various tasks.", {"entities": [(0, 8, "ELEPHANT")]}),
    ("Some elephants have large tusks made of ivory.", {"entities": [(5, 13, "ELEPHANT")]}),
    ("Elephants live in matriarchal herds.", {"entities": [(0, 8, "ELEPHANT")]}),
    ("Certain elephants can live up to 70 years.", {"entities": [(8, 16, "ELEPHANT")]}),
    ("Elephants communicate using low-frequency sounds.", {"entities": [(0, 8, "ELEPHANT")]}),
    ("Some elephants migrate long distances for food and water.", {"entities": [(5, 13, "ELEPHANT")]}),
    ("Elephants have thick skin to protect them from the sun.", {"entities": [(0, 8, "ELEPHANT")]}),
    ("There are three species of elephants in the world.", {"entities": [(29, 37, "ELEPHANT")]}),

    ("Gorillas are intelligent primates living in forests.", {"entities": [(0, 7, "GORILLA")]}),
    ("Do gorillas build nests to sleep in?", {"entities": [(3, 10, "GORILLA")]}),
    ("Gorillas communicate through gestures and vocalizations.", {"entities": [(0, 7, "GORILLA")]}),
    ("Some gorillas have strong family bonds.", {"entities": [(5, 12, "GORILLA")]}),
    ("Gorillas are herbivores, eating leaves and fruits.", {"entities": [(0, 7, "GORILLA")]}),
    ("Certain gorillas have unique facial structures.", {"entities": [(8, 15, "GORILLA")]}),
    ("Gorillas are known for their strength and intelligence.", {"entities": [(0, 7, "GORILLA")]}),
    ("Some gorillas can walk short distances on two legs.", {"entities": [(5, 12, "GORILLA")]}),
    ("Gorillas live in groups led by a dominant male.", {"entities": [(0, 7, "GORILLA")]}),
    ("There are two species of gorillas in the world.", {"entities": [(28, 35, "GORILLA")]}),

    ("Hippos are large semi-aquatic mammals living in rivers.", {"entities": [(0, 5, "HIPPO")]}),
    ("Do hippos spend most of their time in water?", {"entities": [(3, 8, "HIPPO")]}),
    ("Hippos have massive jaws capable of crushing bones.", {"entities": [(0, 5, "HIPPO")]}),
    ("Some hippos can hold their breath for several minutes.", {"entities": [(5, 10, "HIPPO")]}),
    ("Hippos are herbivores, grazing on grasses at night.", {"entities": [(0, 5, "HIPPO")]}),
    ("Certain hippos have unique social behaviors in groups.", {"entities": [(8, 13, "HIPPO")]}),
    ("Hippos are known for their aggressive territorial behavior.", {"entities": [(0, 5, "HIPPO")]}),
    ("Some hippos are more solitary, while others prefer groups.", {"entities": [(5, 10, "HIPPO")]}),
    ("Hippos have thick skin to protect them from the sun.", {"entities": [(0, 5, "HIPPO")]}),
    ("There are two species of hippos in the world.", {"entities": [(29, 34, "HIPPO")]}),

    ("Lizards are fascinating reptiles found in various habitats.", {"entities": [(0, 6, "LIZARD")]}),
    ("Do lizards have the ability to regenerate their tails?", {"entities": [(3, 9, "LIZARD")]}),
    ("Lizards can change their color to blend into surroundings.", {"entities": [(0, 6, "LIZARD")]}),
    ("Some lizards have sticky feet for climbing walls and trees.", {"entities": [(5, 11, "LIZARD")]}),
    ("Lizards are cold-blooded and rely on the sun for warmth.", {"entities": [(0, 6, "LIZARD")]}),
    ("Certain lizards use their tails as a defense mechanism.", {"entities": [(8, 14, "LIZARD")]}),
    ("Lizards communicate through body language and movements.", {"entities": [(0, 6, "LIZARD")]}),
    ("Some lizards are nocturnal hunters that feed on insects.", {"entities": [(5, 11, "LIZARD")]}),
    ("Lizards have excellent vision and detect movement easily.", {"entities": [(0, 6, "LIZARD")]}),
    ("There are over 6,000 species of lizards worldwide.", {"entities": [(31, 37, "LIZARD")]}),

    ("Monkeys are intelligent primates found in tropical forests.", {"entities": [(0, 6, "MONKEY")]}),
    ("Do monkeys use tools to find food?", {"entities": [(3, 9, "MONKEY")]}),
    ("Monkeys can communicate through vocalizations and gestures.", {"entities": [(0, 6, "MONKEY")]}),
    ("Some monkeys have prehensile tails for climbing trees.", {"entities": [(5, 11, "MONKEY")]}),
    ("Monkeys live in social groups and exhibit complex behaviors.", {"entities": [(0, 6, "MONKEY")]}),
    ("Certain monkeys are known for their loud calls.", {"entities": [(8, 14, "MONKEY")]}),
    ("Monkeys are omnivores, eating fruits, insects, and small animals.", {"entities": [(0, 6, "MONKEY")]}),
    ("Some monkeys groom each other to strengthen social bonds.", {"entities": [(5, 11, "MONKEY")]}),
    ("Monkeys have opposable thumbs for grasping objects.", {"entities": [(0, 6, "MONKEY")]}),
    ("There are over 260 species of monkeys worldwide.", {"entities": [(31, 37, "MONKEY")]}),

    ("Mice are small rodents that are found all over the world.", {"entities": [(0, 5, "MOUSE")]}),
    ("Do mice have a good sense of smell?", {"entities": [(3, 7, "MOUSE")]}),
    ("Mice can squeeze through tiny gaps to escape predators.", {"entities": [(0, 5, "MOUSE")]}),
    ("Some mice are used in scientific research.", {"entities": [(5, 9, "MOUSE")]}),
    ("Mice are nocturnal creatures that prefer the dark.", {"entities": [(0, 5, "MOUSE")]}),
    ("Certain mice can jump long distances to find food.", {"entities": [(8, 12, "MOUSE")]}),
    ("Mice communicate using high-pitched squeaks.", {"entities": [(0, 5, "MOUSE")]}),
    ("Some mice live in burrows for protection.", {"entities": [(5, 9, "MOUSE")]}),
    ("Mice have sharp incisors that grow continuously.", {"entities": [(0, 5, "MOUSE")]}),
    ("There are many species of mice around the globe.", {"entities": [(27, 31, "MOUSE")]}),


    ("Pandas are black and white bears native to China.", {"entities": [(0, 5, "PANDA")]}),
    ("Do pandas eat only bamboo?", {"entities": [(3, 8, "PANDA")]}),
    ("Pandas have strong jaws for chewing tough plants.", {"entities": [(0, 5, "PANDA")]}),
    ("Some pandas live in mountain forests.", {"entities": [(5, 10, "PANDA")]}),
    ("Pandas are known for their gentle nature.", {"entities": [(0, 5, "PANDA")]}),
    ("Certain pandas have distinct fur patterns.", {"entities": [(8, 13, "PANDA")]}),
    ("Pandas communicate using vocalizations and scent markings.", {"entities": [(0, 5, "PANDA")]}),
    ("Some pandas can climb trees with ease.", {"entities": [(5, 10, "PANDA")]}),
    ("Pandas spend most of their time eating and sleeping.", {"entities": [(0, 5, "PANDA")]}),
    ("There are few pandas left in the wild.", {"entities": [(14, 19, "PANDA")]}),

    ("Spiders spin intricate webs to catch their prey.", {"entities": [(0, 6, "SPIDER")]}),
    ("Do spiders have eight legs?", {"entities": [(3, 9, "SPIDER")]}),
    ("Spiders are found in nearly every environment.", {"entities": [(0, 6, "SPIDER")]}),
    ("Some spiders use venom to immobilize their prey.", {"entities": [(5, 11, "SPIDER")]}),
    ("Spiders have multiple eyes for detecting movement.", {"entities": [(0, 6, "SPIDER")]}),
    ("Certain spiders can jump long distances to catch insects.", {"entities": [(8, 14, "SPIDER")]}),
    ("Spiders play an important role in controlling insect populations.", {"entities": [(0, 6, "SPIDER")]}),
    ("Some spiders mimic other animals for protection.", {"entities": [(5, 11, "SPIDER")]}),
    ("Spiders can regenerate lost limbs over time.", {"entities": [(0, 6, "SPIDER")]}),
    ("There are thousands of spider species worldwide.", {"entities": [(20, 26, "SPIDER")]}),

    ("Tigers are powerful big cats found in forests and grasslands.", {"entities": [(0, 5, "TIGER")]}),
    ("Do tigers have stripes on their skin?", {"entities": [(3, 9, "TIGER")]}),
    ("Tigers are known for their strength and hunting skills.", {"entities": [(0, 5, "TIGER")]}),
    ("Some tigers are larger than others depending on the species.", {"entities": [(5, 11, "TIGER")]}),
    ("Tigers rely on stealth to catch their prey.", {"entities": [(0, 5, "TIGER")]}),
    ("Certain tigers have adapted to cold climates.", {"entities": [(8, 14, "TIGER")]}),
    ("Tigers are solitary animals except during mating season.", {"entities": [(0, 5, "TIGER")]}),
    ("Some tigers have unique stripe patterns.", {"entities": [(5, 11, "TIGER")]}),
    ("Tigers are an endangered species due to habitat loss.", {"entities": [(0, 5, "TIGER")]}),
    ("There are different subspecies of tigers around the world.", {"entities": [(31, 37, "TIGER")]}),

    ("Zebras have black and white stripes for camouflage.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Do zebras live in herds for protection?", {"entities": [(3, 9, "ZEBRA")]}),
    ("Zebras are closely related to horses and donkeys.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Some zebras migrate long distances for food and water.", {"entities": [(5, 11, "ZEBRA")]}),
    ("Zebras use their stripes to confuse predators.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Certain zebras have unique stripe patterns.", {"entities": [(8, 14, "ZEBRA")]}),
    ("Zebras communicate using vocalizations and body language.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Some zebras can run at high speeds to escape danger.", {"entities": [(5, 11, "ZEBRA")]}),
    ("Zebras are herbivores, grazing on grasses and leaves.", {"entities": [(0, 5, "ZEBRA")]}),
    ("There are three main species of zebras in Africa.", {"entities": [(31, 37, "ZEBRA")]}),
    ("Zebras have black and white stripes for camouflage.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Do zebras live in herds for protection?", {"entities": [(3, 9, "ZEBRA")]}),
    ("Zebras are closely related to horses and donkeys.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Some zebras migrate long distances for food and water.", {"entities": [(5, 11, "ZEBRA")]}),
    ("Zebras use their stripes to confuse predators.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Certain zebras have unique stripe patterns.", {"entities": [(8, 14, "ZEBRA")]}),
    ("Zebras communicate using vocalizations and body language.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Some zebras can run at high speeds to escape danger.", {"entities": [(5, 11, "ZEBRA")]}),
    ("Zebras are herbivores, grazing on grasses and leaves.", {"entities": [(0, 5, "ZEBRA")]}),
    ("There are three main species of zebras in Africa.", {"entities": [(31, 37, "ZEBRA")]}),
    ("Zebras form tight-knit family groups within their herds.", {"entities": [(0, 5, "ZEBRA")]}),
    ("Some zebras have manes that stand upright.", {"entities": [(5, 11, "ZEBRA")]}),
    ("Zebras have strong legs adapted for running from predators.", {"entities": [(0, 5, "ZEBRA")]}),


]


nlp = spacy.load("en_core_web_sm")  


if "ner" not in nlp.pipe_names:
    ner = nlp.add_pipe("ner")
else:
    ner = nlp.get_pipe("ner")


for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])


other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]

with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.create_optimizer()
    for i in range(20):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], losses=losses)
        print(f"Iteration {i+1} Losses: {losses}")

output_dir = "models/ner"
nlp.to_disk(output_dir)
print(f"Model saved to {output_dir}")