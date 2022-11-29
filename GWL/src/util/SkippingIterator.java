package util;

import java.util.Iterator;

public class SkippingIterator<T> implements Iterator<T> {
	
	Iterator<T> it;
	T current;
	T next;
	
	public SkippingIterator(Iterator<T> it) {
		this.it = it;
		this.current = null;
		this.next = getNext();
	}

	public boolean hasNext() {
		return next != null;
	}

	public T next() {
		current = next;
		next = getNext();
		return current;
	}
	public void remove() {
		throw new UnsupportedOperationException();
	}
	
	private T getNext() {
		T next = null;
		while (next == null && it.hasNext()) {
			next = it.next();
		}
		return next;
	}
	
	

}
