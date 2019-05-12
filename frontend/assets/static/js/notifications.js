document.addEventListener('DOMContentLoaded', () => {
	(document.querySelectorAll('.message .close') || []).forEach(($delete) => {
		$notification = $delete.parentNode;
		$delete.addEventListener('click', () => {
		$notification.parentNode.removeChild($notification);
		});
	});
});