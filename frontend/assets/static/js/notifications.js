document.addEventListener('DOMContentLoaded', () => {
	(document.querySelectorAll('.notification .delete') || []).forEach(($delete) => {
		$notification = $delete.parentNode;
		$delete.addEventListener('click', () => {
		$notification.parentNode.removeChild($notification);
		});
	});
});

document.addEventListener('DOMContentLoaded', () => {
	(document.querySelectorAll('.message .close') || []).forEach(($delete) => {
		$notification = $delete.parentNode;
		$delete.addEventListener('click', () => {
		$notification.parentNode.removeChild($notification);
		});
	});
});