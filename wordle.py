import time
from enum import Enum
from string import ascii_lowercase
from typing import Set, List, Dict, Tuple, Callable, Optional
from abc import ABC, abstractmethod


def argmax(ls: List, key: Callable = lambda x: x) -> int:
    return max(range(len(ls)), key=lambda x: key(ls[x]))


class LetterState(Enum):
    CORRECT = 0
    WRONG_PLACE = 1
    NOT_USED = 2


class Wordle(ABC):
    LENGTH = 5
    TRIES = 6

    @abstractmethod
    def check_word(self, word: str) -> Optional[Tuple[List[LetterState], bool]]:
        pass


class InteractiveWordle(Wordle):
    def __init__(self):
        self.input_map: Dict[chr, LetterState] = {
            '.': LetterState.NOT_USED,
            '-': LetterState.WRONG_PLACE,
            '+': LetterState.CORRECT,
        }

    def check_word(self, word: str) -> Optional[Tuple[List[LetterState], bool]]:
        print(f"Try the following: \"{word}\"")
        feedback = input(
            "Give me the input (for each letter: '.' if grey, '-' if yellow, '+' if green, empty if the game is won, \"skip\", to skip the current word)")
        if len(feedback) == 0:
            return [LetterState.CORRECT for _ in word], True
        if feedback == "skip":
            return None
        return [self.input_map[f] for f in feedback], False


class SimulatedWordle(Wordle):
    def __init__(self, word: str, verbose: bool):
        super().__init__()
        assert len(word) == self.LENGTH
        self.word: str = word
        self.verbose = verbose
        self.guesses: List[str] = []
        self.letters: Dict[chr: Set[int]] = {l: set() for l in self.word}
        for i, l in enumerate(self.word):
            self.letters[l].add(i)

    def check_word(self, word: str) -> Optional[Tuple[List[LetterState], bool]]:
        if len(self.guesses) >= self.TRIES:
            print("game is already lost!")
            return [], False
        self.guesses.append(word)
        if word == self.word:
            if self.verbose:
                print("you won!")
            return [LetterState.CORRECT for _ in word], True
        else:
            result = []
            for i, l in enumerate(word):
                if l == self.word[i]:
                    result.append(LetterState.CORRECT)
                elif l in self.word:
                    result.append(LetterState.WRONG_PLACE)
                else:
                    result.append(LetterState.NOT_USED)
            if len(self.guesses) >= self.TRIES and self.verbose:
                print("you lost!")
            return result, False


class Statistics:
    def __init__(self, words: List[str], length: int):
        self.words = words
        self.letter_ranks = self.letter_ranking(self.words)

    @staticmethod
    def get_words(filename: str, length: int) -> List[str]:
        with open(filename, "r") as file:
            words = {line[:length].lower() for line in file if len(line) == length + 1}
            return list(filter(lambda word: all(l in ascii_lowercase for l in word), words))

    @staticmethod
    def letter_ranking(words: List[str]) -> Dict[chr, int]:
        letters = {l: 0 for l in ascii_lowercase}
        for word in words:
            for l in word:
                letters[l] += 1
        return letters

    def score_word(self, word: str, tabu: Set[chr]) -> int:
        letters = set(word) - tabu
        return sum(self.letter_ranks[l] for l in letters)

    def get_best_words(self, tabu: Set[chr]) -> List[str]:
        return sorted(self.words, key=lambda x: self.score_word(x, tabu), reverse=True)

    def get_best_word(self, tabu: Set[chr]) -> str:
        i_best = argmax(self.words, key=lambda x: self.score_word(x, tabu))
        return self.words[i_best]

    @staticmethod
    def filter_words(words: List[str],
                     wrong_words: Set[str],
                     contain_set: Set[chr],
                     fixed_set: Dict[chr, Set[int]],
                     not_contained_set: Set[chr],
                     wrong_positions: Dict[chr, Set[int]]) -> List[str]:

        def meets_requirements(word: str) -> bool:
            return word not in wrong_words and \
                   all(l in word for l in contain_set) and \
                   all(word[i] == l for l, indices in fixed_set.items() for i in indices) and \
                   all(l not in word for l in not_contained_set) and \
                   all(word[i] != l for l, indices in wrong_positions.items() for i in indices)

        return [word for word in words if meets_requirements(word)]


class Solver:
    def __init__(self, wordle: Wordle, words: List[str]):
        self.wordle: Wordle = wordle
        self.stats: Statistics = Statistics(words, self.wordle.LENGTH)
        self.known_letters: Set[chr] = set()
        self.known_not_contained: Set[chr] = set()
        self.known_positions: Dict[chr, Set[int]] = {}
        self.known_wrong_positions: Dict[chr, Set[int]] = {}
        self.known_false_words: Set[str] = set()

    def update(self, word: str, feedback: List[LetterState]):
        self.known_false_words.add(word)
        for i, l in enumerate(word):
            if feedback[i] == LetterState.NOT_USED:
                self.known_not_contained.add(l)
            else:
                self.known_letters.add(l)
                if feedback[i] == LetterState.CORRECT:
                    if l not in self.known_positions:
                        self.known_positions[l] = {i}
                    else:
                        self.known_positions[l].add(i)
                else:
                    assert (feedback[i] == LetterState.WRONG_PLACE)
                    if l not in self.known_wrong_positions:
                        self.known_wrong_positions[l] = {i}
                    else:
                        self.known_wrong_positions[l].add(i)

    def solve(self) -> bool:
        guessed = []
        possible_words = self.stats.words

        for iteration in range(self.wordle.TRIES):

            possible_words = Statistics.filter_words(
                possible_words, self.known_false_words,
                self.known_letters, self.known_positions, self.known_not_contained, self.known_wrong_positions)

            remaining_tries = self.wordle.TRIES - iteration

            if len(possible_words) <= remaining_tries or \
                    remaining_tries <= 3:
                # try to guess the word
                assert len(possible_words) > 0
                stats = Statistics(possible_words, self.wordle.LENGTH)
                word = stats.get_best_word(self.known_letters | self.known_not_contained)
            else:
                # get more letters
                word = self.stats.get_best_word(self.known_letters | self.known_not_contained)

            guessed.append(word)
            response = self.wordle.check_word(word)
            if response is None:
                pass  # TODO pick another word
            feedback, won = response
            if won:
                return True
            else:
                self.update(word, feedback)
        if isinstance(self.wordle, SimulatedWordle):
            print(f"{self.wordle.word} not found, remaining words: {possible_words[:10]}...({len(possible_words)})")
            print(f"guessed: {guessed}")
        return False


def evaluate_algorithm(filename: str, verbose: bool):
    wordlist = Statistics.get_words(filename, Wordle.LENGTH)
    print(f"number of words: {len(wordlist)}")

    not_solved = 0

    t_start = time.time()

    for i, word in enumerate(wordlist):
        wordle = SimulatedWordle(word, False)
        solver = Solver(wordle, wordlist)
        if not solver.solve():
            not_solved += 1
        if (i + 1) % 50 == 0 and verbose:
            print(f"currently working at {(i + 1) / (time.time() - t_start):.1f} words per second")
    print(f"Found {len(wordlist) - not_solved} of {len(wordlist)} words, failed at {not_solved}")


def main():
    # evaluate_algorithm("deutsch.txt", False)
    wordlist = Statistics.get_words("deutsch.txt", Wordle.LENGTH)
    print("drama" in wordlist
          )
    wordle = InteractiveWordle()
    solver = Solver(wordle, wordlist)
    solver.solve()


if __name__ == '__main__':
    main()
