package me.soulmachine;

import org.apache.hadoop.io.Text;

public class WordBest implements Comparable<WordBest> {
	int count;
	Text word;
	public WordBest(Text word2, int sum) {
		this.word = word2; this.count = sum;
	}
	@Override
	public int compareTo(WordBest arg0) {
		return arg0.count - count;
	}
	
	@Override 
	public String toString() {
		return word.toString();
	}
}