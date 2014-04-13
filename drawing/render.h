#ifndef RENDER_H
#define RENDER_H

#include <QWidget>
#include <QBrush>
#include <QPen>
#include <QPixmap>

class render : public QWidget
{
    Q_OBJECT

public:
    explicit render(QWidget *parent = 0);
    QSize minimumSizeHint() const;
    QSize sizeHint() const;

signals:

public slots:

private:
    QPen pen;
    QBrush brush;
};

#endif // RENDER_H
