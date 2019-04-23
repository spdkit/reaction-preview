install: stow
	stow --verbose --adopt --no-folding --target ~/.local pkg
uninstall:
	stow --verbose --target ~/.local --delete pkg
stow:
	which stow
